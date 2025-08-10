/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
 * Middletown, DE 19709, New Castle County, USA.
 *
 */

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>

#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringMap.h"
#include "toolchain/Support/Casting.h"
#include "toolchain/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/TypeUtilities.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_device.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_types.h"
#include "machina/compiler/mlir/machina/transforms/collection_ops_util.h"
#include "machina/compiler/mlir/machina/transforms/passes.h"
#include "machina/compiler/mlir/machina/utils/convert_tensor.h"
#include "machina/compiler/mlir/machina/utils/dynamic_shape_utils.h"
#include "machina/compiler/mlir/machina/utils/mangling_util.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/types.h"

namespace mlir {

namespace {

namespace cutil = TF::collection_ops_util;

#define GEN_PASS_DEF_TENSORLISTOPSDECOMPOSITIONPASS
#include "machina/compiler/mlir/machina/transforms/tf_passes.h.inc"

struct TensorListOpsDecompositionPass
    : public impl::TensorListOpsDecompositionPassBase<
          TensorListOpsDecompositionPass> {
  void runOnOperation() override;
};

// Updates func's type according to its current arguments and return values.
void UpdateFuncType(func::FuncOp func) {
  toolchain::SmallVector<Type, 8> arg_types;
  for (auto arg : func.getArguments()) arg_types.push_back(arg.getType());
  func.setType(
      FunctionType::get(func.getContext(), arg_types,
                        func.front().getTerminator()->getOperandTypes()));
}

// Holds the size value of a tensor list and whether the size is statically
// known (fixed).
struct SizeInfo {
  Value size;
  bool fixed;
};

// Modifies a function's signature to rewrite tensor list arguments to buffers
// and sizes.
void ModifyFunctionSignature(
    func::FuncOp func, Type size_type,
    toolchain::SmallDenseMap<Value, SizeInfo>* buffer_to_size,
    toolchain::function_ref<std::optional<Type>(int64_t)> arg_to_buffer_type,
    toolchain::function_ref<bool(int64_t)> arg_buffer_size_is_fixed) {
  auto new_input_types = toolchain::to_vector<8>(func.getFunctionType().getInputs());
  int64_t original_arg_count = new_input_types.size();
  Location loc = func.getLoc();
  for (int64_t i = 0; i < original_arg_count; ++i) {
    auto buffer_type = arg_to_buffer_type(i);
    if (!buffer_type.has_value()) continue;
    func.getArgument(i).setType(*buffer_type);
    new_input_types[i] = *buffer_type;
    auto size_arg = func.front().addArgument(size_type, loc);
    new_input_types.push_back(size_arg.getType());
    if (buffer_to_size) {
      (*buffer_to_size)[func.getArgument(i)] = {size_arg,
                                                arg_buffer_size_is_fixed(i)};
    }
  }
  UpdateFuncType(func);
}

// Holds information about a decomposed callee function for
// PartitionedCall/StatefulPartitionedCall.
struct PartitionedCallDecompositionInfo {
  bool signature_change;
  func::FuncOp decomposed_callee;
  toolchain::SmallDenseMap<int64_t, int64_t> buffer_arg_to_size_arg;
  // Each element is a tuple of (buffer_return_index, size_return_index,
  // fixed_size).
  toolchain::SmallVector<std::tuple<int64_t, int64_t, bool>, 8>
      buffer_ret_to_size_ret;
};

LogicalResult DecomposeTensorListOpsInternal(
    Block*, ModuleOp, toolchain::SmallDenseMap<Value, SizeInfo>*,
    toolchain::StringMap<PartitionedCallDecompositionInfo>*);

// Adds the corresponding sizes of tensor list buffers in block's terminator
// to the list of return values. Returns the mapping from the buffer
// indices to the added size indices, which is a list of tuples
// (buffer_return_index, size_return_index, fixed_size).
template <class TerminatorOp>
toolchain::SmallVector<std::tuple<int64_t, int64_t, bool>, 8>
AddTensorListSizesToTerminator(
    Block& block, const toolchain::SmallDenseMap<Value, SizeInfo>& buffer_to_size) {
  auto old_terminator = block.getTerminator();
  auto new_outputs = toolchain::to_vector<8>(old_terminator->getOperands());
  toolchain::SmallVector<std::tuple<int64_t, int64_t, bool>, 8>
      output_buffer_to_size;
  for (auto retval : toolchain::enumerate(old_terminator->getOperands())) {
    auto it = buffer_to_size.find(retval.value());
    if (it == buffer_to_size.end()) continue;
    output_buffer_to_size.emplace_back(retval.index(), new_outputs.size(),
                                       it->getSecond().fixed);
    new_outputs.push_back(it->getSecond().size);
  }
  OpBuilder builder(old_terminator);
  TerminatorOp::create(builder, old_terminator->getLoc(), new_outputs);
  old_terminator->erase();
  return output_buffer_to_size;
}

// Adds the corresponding sizes of tensor list buffers in func's return values
// to the list of return values. Returns the mapping from the buffer indices to
// the added size indices, which is a list of tuples (buffer_return_index,
// size_return_index, fixed_size).
toolchain::SmallVector<std::tuple<int64_t, int64_t, bool>, 8> ModifyFunctionReturn(
    func::FuncOp func,
    const toolchain::SmallDenseMap<Value, SizeInfo>& buffer_to_size) {
  auto output_buffer_to_size = AddTensorListSizesToTerminator<func::ReturnOp>(
      func.front(), buffer_to_size);
  UpdateFuncType(func);
  return output_buffer_to_size;
}

LogicalResult HandleWhileOp(
    TF::WhileOp while_op, ModuleOp module,
    toolchain::SmallDenseMap<Value, SizeInfo>* buffer_to_size,
    toolchain::StringMap<PartitionedCallDecompositionInfo>*
        decomposed_partitioned_call_callees) {
  // Rewrite body.
  auto body = while_op.body_function();
  toolchain::SmallDenseMap<Value, SizeInfo> body_map;
  auto find_arg_tensor_list_type = [&](int64_t index) -> std::optional<Type> {
    auto it = buffer_to_size->find(while_op.getOperand(index));
    if (it == buffer_to_size->end()) return std::nullopt;
    return it->getFirst().getType();
  };
  auto arg_buffer_size_is_fixed = [&](int64_t index) {
    return (*buffer_to_size)[while_op.getOperand(index)].fixed;
  };
  OpBuilder builder(while_op);
  ModifyFunctionSignature(body, cutil::GetSizeType(builder), &body_map,
                          find_arg_tensor_list_type, arg_buffer_size_is_fixed);
  if (failed(DecomposeTensorListOpsInternal(
          &body.front(), module, &body_map,
          decomposed_partitioned_call_callees))) {
    return failure();
  }
  auto output_buffer_to_size = ModifyFunctionReturn(body, body_map);

  // Rewrite cond.
  auto cond = while_op.cond_function();
  toolchain::SmallDenseMap<Value, SizeInfo> cond_map;
  ModifyFunctionSignature(cond, cutil::GetSizeType(builder), &cond_map,
                          find_arg_tensor_list_type, arg_buffer_size_is_fixed);
  if (failed(DecomposeTensorListOpsInternal(
          &cond.front(), module, &cond_map,
          decomposed_partitioned_call_callees))) {
    return failure();
  }
  if (output_buffer_to_size.empty()) {
    return success();
  }
  // Create the new while op.
  auto new_while_operands = toolchain::to_vector<8>(while_op.getOperands());
  for (int64_t i = 0; i < while_op.getNumResults(); ++i) {
    auto it = buffer_to_size->find(while_op.getOperand(i));
    if (it == buffer_to_size->end()) continue;
    new_while_operands.push_back(it->getSecond().size);
  }
  auto new_while = TF::WhileOp::create(
      builder, while_op.getLoc(), body.getFunctionType().getInputs(),
      new_while_operands, while_op->getAttrs());
  for (const auto& entry : output_buffer_to_size) {
    (*buffer_to_size)[new_while.getResult(std::get<0>(entry))] = {
        new_while.getResult(std::get<1>(entry)), std::get<2>(entry)};
  }
  while_op.replaceAllUsesWith(
      new_while.getResults().take_front(while_op.getNumResults()));
  while_op.erase();
  return success();
}

template <class CaseOrIfOp>
LogicalResult HandleCaseOrIfOp(
    CaseOrIfOp op, ArrayRef<func::FuncOp> branches, ModuleOp module,
    toolchain::SmallDenseMap<Value, SizeInfo>* buffer_to_size,
    toolchain::StringMap<PartitionedCallDecompositionInfo>*
        decomposed_partitioned_call_callees) {
  // Rewrite the branches.
  SmallVector<toolchain::SmallDenseMap<Value, SizeInfo>, 2> branch_maps;
  branch_maps.resize(branches.size());

  auto find_arg_buffer_type = [&](int64_t index) -> std::optional<Type> {
    auto it = buffer_to_size->find(op.getOperand(index + 1));
    if (it == buffer_to_size->end()) return std::nullopt;
    return it->getFirst().getType();
  };
  auto arg_buffer_size_is_fixed = [&](int64_t index) {
    return (*buffer_to_size)[op.getOperand(index + 1)].fixed;
  };
  OpBuilder builder(op);
  for (const auto& pair : toolchain::zip(branches, branch_maps)) {
    func::FuncOp branch = std::get<0>(pair);
    toolchain::SmallDenseMap<Value, SizeInfo>& branch_map = std::get<1>(pair);
    ModifyFunctionSignature(branch, cutil::GetSizeType(builder), &branch_map,
                            find_arg_buffer_type, arg_buffer_size_is_fixed);

    if (failed(DecomposeTensorListOpsInternal(
            &branch.front(), module, &branch_map,
            decomposed_partitioned_call_callees)))
      return failure();
  }

  const bool arg_no_changed = branch_maps.front().empty();
  auto output_buffer_to_size =
      ModifyFunctionReturn(branches.front(), branch_maps.front());
  for (const auto& pair : toolchain::drop_begin(toolchain::zip(branches, branch_maps), 1))
    ModifyFunctionReturn(std::get<0>(pair), std::get<1>(pair));

  if (output_buffer_to_size.empty() && arg_no_changed) return success();

  // Recreate the op.
  auto new_operands = toolchain::to_vector<8>(op.getOperands());
  for (int64_t i = 1; i < op.getNumOperands(); ++i) {
    auto it = buffer_to_size->find(op.getOperand(i));
    if (it == buffer_to_size->end()) continue;
    new_operands.push_back(it->getSecond().size);
  }
  func::FuncOp first_branch = branches.front();
  builder.setInsertionPoint(op);
  auto new_op = CaseOrIfOp::create(builder, op.getLoc(),
                                   first_branch.getFunctionType().getResults(),
                                   new_operands, op->getAttrs());
  for (const auto& entry : output_buffer_to_size) {
    (*buffer_to_size)[new_op.getResult(std::get<0>(entry))] = {
        new_op.getResult(std::get<1>(entry)), std::get<2>(entry)};
  }
  op.replaceAllUsesWith(new_op.getResults().take_front(op.getNumResults()));
  op.erase();
  return success();
}

LogicalResult HandleWhileRegionOp(
    TF::WhileRegionOp while_op, ModuleOp module,
    toolchain::SmallDenseMap<Value, SizeInfo>* buffer_to_size,
    toolchain::StringMap<PartitionedCallDecompositionInfo>*
        decomposed_partitioned_call_callees) {
  OpBuilder builder(while_op);
  auto modify_region_arguments = [&](Region& region) {
    int64_t original_arg_count = region.getNumArguments();
    for (int64_t i = 0; i < original_arg_count; ++i) {
      auto operand = while_op.getOperand(i);
      auto it = buffer_to_size->find(operand);
      if (it == buffer_to_size->end()) continue;
      auto buffer_type = it->getFirst().getType();
      region.getArgument(i).setType(buffer_type);
      auto size_arg =
          region.addArgument(cutil::GetSizeType(builder), region.getLoc());
      (*buffer_to_size)[region.getArgument(i)] = {size_arg,
                                                  it->getSecond().fixed};
    }
  };

  // Rewrite body.
  Region& body_region = while_op.getBody();
  modify_region_arguments(body_region);
  if (failed(DecomposeTensorListOpsInternal(
          &body_region.front(), module, buffer_to_size,
          decomposed_partitioned_call_callees))) {
    return failure();
  }
  auto output_buffer_to_size = AddTensorListSizesToTerminator<TF::YieldOp>(
      body_region.front(), *buffer_to_size);

  // Rewrite cond.
  Region& cond_region = while_op.getCond();
  modify_region_arguments(cond_region);
  if (failed(DecomposeTensorListOpsInternal(
          &cond_region.front(), module, buffer_to_size,
          decomposed_partitioned_call_callees))) {
    return failure();
  }

  if (output_buffer_to_size.empty()) return success();

  // Create the new while op.
  auto new_while_operands = toolchain::to_vector<8>(while_op.getOperands());
  for (int64_t i = 0; i < while_op.getNumResults(); ++i) {
    auto it = buffer_to_size->find(while_op.getOperand(i));
    if (it == buffer_to_size->end()) continue;
    new_while_operands.push_back(it->getSecond().size);
  }
  auto new_while = TF::WhileRegionOp::create(
      builder, while_op.getLoc(),
      body_region.front().getTerminator()->getOperandTypes(),
      new_while_operands, while_op->getAttrs());
  new_while.getBody().takeBody(body_region);
  new_while.getCond().takeBody(cond_region);
  for (const auto& entry : output_buffer_to_size) {
    (*buffer_to_size)[new_while.getResult(std::get<0>(entry))] = {
        new_while.getResult(std::get<1>(entry)), std::get<2>(entry)};
  }
  while_op.replaceAllUsesWith(
      new_while.getResults().take_front(while_op.getNumResults()));
  while_op.erase();
  return success();
}

LogicalResult HandleIfRegionOp(
    TF::IfRegionOp if_op, ModuleOp module,
    toolchain::SmallDenseMap<Value, SizeInfo>* buffer_to_size,
    toolchain::StringMap<PartitionedCallDecompositionInfo>*
        decomposed_partitioned_call_callees) {
  // Rewrite the branches.
  Region& then_branch = if_op.getThenBranch();
  Region& else_branch = if_op.getElseBranch();
  if (failed(DecomposeTensorListOpsInternal(
          &then_branch.front(), module, buffer_to_size,
          decomposed_partitioned_call_callees)))
    return failure();
  if (failed(DecomposeTensorListOpsInternal(
          &else_branch.front(), module, buffer_to_size,
          decomposed_partitioned_call_callees)))
    return failure();

  auto output_buffer_to_size = AddTensorListSizesToTerminator<TF::YieldOp>(
      then_branch.front(), *buffer_to_size);
  AddTensorListSizesToTerminator<TF::YieldOp>(else_branch.front(),
                                              *buffer_to_size);

  if (output_buffer_to_size.empty()) return success();

  // Recreate the op.
  OpBuilder builder(if_op);
  auto new_op = TF::IfRegionOp::create(
      builder, if_op.getLoc(),
      then_branch.front().getTerminator()->getOperandTypes(),
      if_op.getOperand(), if_op->getAttrs());
  for (const auto& entry : output_buffer_to_size) {
    (*buffer_to_size)[new_op.getResult(std::get<0>(entry))] = {
        new_op.getResult(std::get<1>(entry)), std::get<2>(entry)};
  }

  new_op.getThenBranch().takeBody(if_op.getThenBranch());
  new_op.getElseBranch().takeBody(if_op.getElseBranch());

  if_op.replaceAllUsesWith(
      new_op.getResults().take_front(if_op.getNumResults()));
  if_op.erase();
  return success();
}

LogicalResult HandleCaseRegionOp(
    TF::CaseRegionOp case_op, ModuleOp module,
    toolchain::SmallDenseMap<Value, SizeInfo>* buffer_to_size,
    toolchain::StringMap<PartitionedCallDecompositionInfo>*
        decomposed_partitioned_call_callees) {
  // Rewrite the branches.
  RegionRange branches = case_op.getRegions();

  for (Region* branch : branches) {
    if (failed(DecomposeTensorListOpsInternal(
            &branch->front(), module, buffer_to_size,
            decomposed_partitioned_call_callees)))
      return failure();
  }

  // Get the output buffer index to size index mapping one of the branches. It
  // should be same for all the branches so we only get it for the first branch.
  Region* first_branch = branches.front();
  auto output_buffer_to_size = AddTensorListSizesToTerminator<TF::YieldOp>(
      first_branch->front(), *buffer_to_size);
  for (Region* branch : branches.drop_front()) {
    AddTensorListSizesToTerminator<TF::YieldOp>(branch->front(),
                                                *buffer_to_size);
  }

  if (output_buffer_to_size.empty()) return success();

  // Recreate the op.
  OpBuilder builder(case_op);
  auto new_op = TF::CaseRegionOp::create(
      builder, case_op.getLoc(),
      first_branch->front().getTerminator()->getOperandTypes(),
      case_op.getOperand(), case_op->getAttrs(), case_op.getNumRegions());
  for (const auto& entry : output_buffer_to_size) {
    (*buffer_to_size)[new_op.getResult(std::get<0>(entry))] = {
        new_op.getResult(std::get<1>(entry)), std::get<2>(entry)};
  }

  for (auto pair : toolchain::zip(new_op.getRegions(), case_op.getRegions())) {
    std::get<0>(pair)->takeBody(*std::get<1>(pair));
  }
  case_op.replaceAllUsesWith(
      new_op.getResults().take_front(case_op.getNumResults()));
  case_op.erase();
  return success();
}

template <typename CallOp>
LogicalResult HandlePartitionedCallOp(
    CallOp call, func::FuncOp callee, ModuleOp module,
    toolchain::SmallDenseMap<Value, SizeInfo>* buffer_to_size,
    toolchain::StringMap<PartitionedCallDecompositionInfo>*
        decomposed_partitioned_call_callees) {
  auto emplace_res = decomposed_partitioned_call_callees->try_emplace(
      callee.getName(), PartitionedCallDecompositionInfo());
  auto& info = emplace_res.first->second;
  // Recreates the call op with info.
  auto recreate_caller = [&] {
    auto new_operands = toolchain::to_vector<8>(call.getOperands());
    for (int64_t i = 0; i < call.getNumOperands(); ++i) {
      auto arg_it = info.buffer_arg_to_size_arg.find(i);
      if (arg_it == info.buffer_arg_to_size_arg.end()) continue;
      auto it = buffer_to_size->find(call.getOperand(i));
      if (it == buffer_to_size->end()) {
        call.emitOpError("unknown tensor list.");
        return failure();
      }
      assert(arg_it->second == new_operands.size());
      new_operands.push_back(it->getSecond().size);
    }
    OpBuilder builder(call);
    auto new_call =
        CallOp::create(builder, call.getLoc(),
                       info.decomposed_callee.getFunctionType().getResults(),
                       new_operands, call->getAttrs());
    new_call->setAttr(
        "f", SymbolRefAttr::get(
                 builder.getContext(),
                 const_cast<func::FuncOp&>(info.decomposed_callee).getName()));
    for (const auto& entry : info.buffer_ret_to_size_ret) {
      (*buffer_to_size)[new_call.getResult(std::get<0>(entry))] = {
          new_call.getResult(std::get<1>(entry)), std::get<2>(entry)};
    }
    call.replaceAllUsesWith(
        new_call.getResults().take_front(call.getNumResults()));
    call.erase();
    return success();
  };
  if (!emplace_res.second) {
    // This callee was handled before.
    if (!info.signature_change) return success();
    return recreate_caller();
  }
  // Rewrite the callee.
  toolchain::SmallDenseMap<Value, SizeInfo> callee_map;
  func::FuncOp lowered_callee = callee;
  if (!callee.isPrivate()) {
    // Clone non-private callee in case of signature change.
    lowered_callee = callee.clone();
    lowered_callee.setPrivate();
  }
  auto find_arg_buffer_type = [&](int64_t index) -> std::optional<Type> {
    auto it = buffer_to_size->find(call.getOperand(index));
    if (it == buffer_to_size->end()) return std::nullopt;
    return it->getFirst().getType();
  };
  auto arg_buffer_size_is_fixed = [&](int64_t index) {
    return (*buffer_to_size)[call.getOperand(index)].fixed;
  };
  ModifyFunctionSignature(lowered_callee, cutil::GetSizeType(OpBuilder(call)),
                          &callee_map, find_arg_buffer_type,
                          arg_buffer_size_is_fixed);
  const bool args_no_changed = callee_map.empty();
  if (failed(DecomposeTensorListOpsInternal(
          &lowered_callee.front(), module, &callee_map,
          decomposed_partitioned_call_callees))) {
    return failure();
  }
  info.buffer_ret_to_size_ret =
      ModifyFunctionReturn(lowered_callee, callee_map);
  info.decomposed_callee = lowered_callee;
  if (args_no_changed && info.buffer_ret_to_size_ret.empty()) {
    // Signature is not modified. We do not need to keep two copies.
    info.signature_change = false;
    if (lowered_callee != callee) {
      lowered_callee.setName(
          StringAttr::get(callee->getContext(), callee.getName()));
      callee.erase();
      SymbolTable(module).insert(lowered_callee);
    }
  } else {
    info.signature_change = true;
    for (auto& entry : callee_map) {
      auto buffer_arg = mlir::dyn_cast<BlockArgument>(entry.getFirst());
      if (!buffer_arg) continue;
      info.buffer_arg_to_size_arg[buffer_arg.getArgNumber()] =
          mlir::cast<BlockArgument>(entry.getSecond().size).getArgNumber();
    }
    if (lowered_callee != callee) {
      // Add the clone with a new name.
      lowered_callee.setName(StringAttr::get(
          callee->getContext(),
          toolchain::formatv("{0}_tensorlist_decomposed", callee.getName()).str()));
      SymbolTable(module).insert(lowered_callee);
      callee = lowered_callee;
    }
  }
  if (info.signature_change) return recreate_caller();
  return success();
}

// Parses an R1 value to `shape` if it is a TF::ConstOp output. Otherwise,
// returns an error.
LogicalResult GetConstShapeValue(Value shape_value,
                                 toolchain::SmallVector<int64_t, 8>* shape) {
  auto shape_op = shape_value.getDefiningOp();
  if (!shape_op) return failure();
  auto shape_const_op = toolchain::dyn_cast<TF::ConstOp>(shape_op);
  if (!shape_const_op) return failure();
  for (const auto& v : shape_const_op.getValue().getValues<APInt>()) {
    int64_t dim_size = v.getSExtValue();
    if (dim_size == machina::kTFDynamicSize) return failure();
    shape->push_back(dim_size);
  }
  return success();
}

// Checks the result Variant type to infer the element shape if fully defined.
// If the Variant type has multiple subtypes or does not have static shape,
// return error.
LogicalResult GetElementShapeFromResultType(
    Type type, toolchain::SmallVector<int64_t, 8>* shape) {
  auto variant_type =
      mlir::dyn_cast<TF::VariantType>(getElementTypeOrSelf(type));
  if (!variant_type || variant_type.getSubtypes().size() != 1) return failure();
  TensorType tensor_type = variant_type.getSubtypes().front();
  if (!tensor_type.hasStaticShape()) return failure();
  for (auto d : tensor_type.getShape()) shape->push_back(d);
  return success();
}

LogicalResult HandleEmptyTensorListOp(
    TF::EmptyTensorListOp list,
    toolchain::SmallDenseMap<Value, SizeInfo>* buffer_to_size) {
  Value buffer;
  OpBuilder builder(list);
  toolchain::SmallVector<int64_t, 8> element_shape;
  // Infer TensorList element shape from the return type first, and then from
  // the const element shape operand. We first check the return type because
  // shape inference might have successfully inferred the element shape from
  // write operations on the TensorList.
  if (failed(GetElementShapeFromResultType(list.getType(), &element_shape))) {
    if (failed(GetConstShapeValue(list.getElementShape(), &element_shape))) {
      return list.emitOpError("unknown tensor list element shape");
    }
  }
  if (failed(cutil::CreateInitBufferValue(
          element_shape, list.getMaxNumElements(), list, list.getElementDtype(),
          builder, &buffer))) {
    return failure();
  }
  Value size = cutil::GetR1Const({0LL}, builder, list.getLoc());
  list.getHandle().replaceAllUsesWith(buffer);
  (*buffer_to_size)[buffer] = {size, /*fixed=*/false};
  list.erase();
  return success();
}

LogicalResult HandleTensorListReserveOp(
    TF::TensorListReserveOp list,
    toolchain::SmallDenseMap<Value, SizeInfo>* buffer_to_size) {
  Value buffer;
  OpBuilder builder(list);
  toolchain::SmallVector<int64_t, 8> element_shape;
  // Infer TensorList element shape from the return type first, and then from
  // the const element shape operand. We first check the return type because
  // shape inference might have successfully inferred the element shape from
  // write operations on the TensorList.
  if (failed(GetElementShapeFromResultType(list.getType(), &element_shape))) {
    if (failed(GetConstShapeValue(list.getElementShape(), &element_shape))) {
      return list.emitOpError("unknown tensor list element shape");
    }
  }
  if (failed(cutil::CreateInitBufferValue(element_shape, list.getNumElements(),
                                          list, list.getElementDtype(), builder,
                                          &buffer))) {
    return failure();
  }
  Value size = cutil::ReshapeScalarToSizeType(builder, list.getNumElements(),
                                              list.getLoc());
  (*buffer_to_size)[buffer] = {size, /*fixed=*/true};
  list.getHandle().replaceAllUsesWith(buffer);
  list.erase();
  return success();
}

LogicalResult HandleTensorListFromTensorOp(
    TF::TensorListFromTensorOp list,
    toolchain::SmallDenseMap<Value, SizeInfo>* buffer_to_size) {
  OpBuilder builder(list);
  Value buffer = TF::IdentityOp::create(
      builder, list.getLoc(), ArrayRef<Type>{list.getTensor().getType()},
      ArrayRef<Value>{list.getTensor()});
  auto type = mlir::cast<TensorType>(buffer.getType());
  if (!type.hasStaticShape()) {
    return list.emitOpError("TensorListFromTensorOp input has unknown shape.");
  }
  Value size = cutil::GetR1Const({type.getShape()[0]}, builder, list.getLoc());
  (*buffer_to_size)[buffer] = {size, /*fixed=*/true};
  list.getOutputHandle().replaceAllUsesWith(buffer);
  list.erase();
  return success();
}

LogicalResult HandleTensorListPushBackOp(
    TF::TensorListPushBackOp push,
    toolchain::SmallDenseMap<Value, SizeInfo>* buffer_to_size) {
  auto buffer = push.getInputHandle();
  auto it = buffer_to_size->find(buffer);
  if (it == buffer_to_size->end()) {
    return push.emitOpError(
        "found tf.TensorListPushBack on unknown TensorList.");
  }
  if (it->getSecond().fixed) {
    return push.emitError("cannot push on a fixed-size tensor list");
  }
  auto size = it->getSecond().size;
  OpBuilder builder(push);
  auto new_buffer =
      cutil::SetElement(size, buffer, push.getTensor(), builder, push.getLoc());
  auto new_size = TF::AddV2Op::create(
      builder, push.getLoc(), ArrayRef<Type>{size.getType()},
      ArrayRef<Value>{size, cutil::GetR1Const({1LL}, builder, push.getLoc())});
  push.getOutputHandle().replaceAllUsesWith(new_buffer);
  (*buffer_to_size)[new_buffer] = {new_size, /*fixed=*/false};
  push.erase();
  return success();
}

LogicalResult HandleTensorListPopBackOp(
    TF::TensorListPopBackOp pop,
    toolchain::SmallDenseMap<Value, SizeInfo>* buffer_to_size) {
  auto buffer = pop.getInputHandle();
  auto it = buffer_to_size->find(buffer);
  if (it == buffer_to_size->end()) {
    pop.emitOpError("found tf.TensorListPopBack on unknown TensorList.");
    return failure();
  }
  if (it->getSecond().fixed) {
    return pop.emitError("cannot pop on a fixed-size tensor list");
  }
  auto size = it->getSecond().size;
  OpBuilder builder(pop);
  auto new_buffer = TF::IdentityOp::create(builder, pop.getLoc(),
                                           ArrayRef<Type>{buffer.getType()},
                                           ArrayRef<Value>{buffer});
  auto new_size = TF::SubOp::create(
      builder, pop.getLoc(), ArrayRef<Type>{size.getType()},
      ArrayRef<Value>{size, cutil::GetR1Const({1LL}, builder, pop.getLoc())});
  auto element = cutil::GetElement(new_size, new_buffer, builder, pop.getLoc());
  pop.getOutputHandle().replaceAllUsesWith(new_buffer);
  pop.getTensor().replaceAllUsesWith(element);
  pop.erase();
  (*buffer_to_size)[new_buffer] = {new_size, /*fixed=*/false};
  return success();
}

LogicalResult HandleTensorListGetItemOp(
    TF::TensorListGetItemOp get_item,
    const toolchain::SmallDenseMap<Value, SizeInfo>& buffer_to_size) {
  auto buffer = get_item.getInputHandle();
  auto it = buffer_to_size.find(buffer);
  if (it == buffer_to_size.end()) {
    get_item.emitOpError("found tf.TensorListGetItemOp on unknown TensorList.");
    return failure();
  }
  OpBuilder builder(get_item);
  auto index = cutil::ReshapeScalarToSizeType(builder, get_item.getIndex(),
                                              get_item.getLoc());
  auto element =
      cutil::GetElement(index, buffer, OpBuilder(get_item), get_item.getLoc());
  get_item.getItem().replaceAllUsesWith(element);
  get_item.erase();
  return success();
}

LogicalResult HandleTensorListSetItemOp(
    TF::TensorListSetItemOp set_item,
    toolchain::SmallDenseMap<Value, SizeInfo>* buffer_to_size) {
  auto buffer = set_item.getInputHandle();
  auto it = buffer_to_size->find(buffer);
  if (it == buffer_to_size->end()) {
    set_item.emitOpError("found tf.TensorListSetItemOp on unknown TensorList.");
    return failure();
  }
  OpBuilder builder(set_item);
  auto index = cutil::ReshapeScalarToSizeType(builder, set_item.getIndex(),
                                              set_item.getLoc());
  auto new_buffer = cutil::SetElement(index, buffer, set_item.getItem(),
                                      builder, set_item.getLoc());
  set_item.getOutputHandle().replaceAllUsesWith(new_buffer);
  auto size = it->getSecond();
  (*buffer_to_size)[new_buffer] = size;
  set_item.erase();
  return success();
}

LogicalResult HandleTensorListLengthOp(
    TF::TensorListLengthOp length,
    const toolchain::SmallDenseMap<Value, SizeInfo>& buffer_to_size) {
  auto it = buffer_to_size.find(length.getInputHandle());
  if (it == buffer_to_size.end()) {
    length.emitOpError("found tf.TensorListLength on unknown TensorList.");
    return failure();
  }
  OpBuilder builder(length);
  if (it->getSecond().fixed) {
    auto dim = cutil::CreateScalarConst(
        mlir::cast<RankedTensorType>(length.getInputHandle().getType())
            .getDimSize(0),
        builder, length.getLoc());
    length.getLength().replaceAllUsesWith(dim);
  } else {
    auto current_size = it->getSecond().size;
    // Reshapes the R1 length to a scalar.
    auto reshape = TF::ReshapeOp::create(
        builder, length.getLoc(),
        ArrayRef<Type>{RankedTensorType::get(
            {}, getElementTypeOrSelf(current_size.getType()))},
        ArrayRef<Value>{current_size,
                        cutil::GetR1Const({}, builder, length.getLoc())});
    length.getLength().replaceAllUsesWith(reshape);
  }
  length.erase();
  return success();
}

LogicalResult HandleTensorListElementShapeOp(
    TF::TensorListElementShapeOp elem_shape,
    const toolchain::SmallDenseMap<Value, SizeInfo>& buffer_to_size) {
  if (buffer_to_size.count(elem_shape.getInputHandle()) == 0) {
    return elem_shape.emitOpError("unknown tensor list");
  }
  auto buffer = elem_shape.getInputHandle();
  auto result = cutil::GetR1Const(
      mlir::cast<RankedTensorType>(buffer.getType()).getShape().drop_front(),
      OpBuilder(elem_shape), elem_shape.getLoc(),
      elem_shape.getShapeType().getIntOrFloatBitWidth());
  elem_shape.getElementShape().replaceAllUsesWith(result);
  elem_shape.erase();
  return success();
}

LogicalResult HandleTensorListGatherOp(
    TF::TensorListGatherOp gather,
    const toolchain::SmallDenseMap<Value, SizeInfo>& buffer_to_size) {
  auto it = buffer_to_size.find(gather.getInputHandle());
  if (it == buffer_to_size.end()) {
    return gather.emitOpError("unknown tensor list");
  }
  auto buffer = gather.getInputHandle();
  auto result = cutil::GatherElements(gather.getIndices(), buffer,
                                      OpBuilder(gather), gather.getLoc());
  gather.getValues().replaceAllUsesWith(result);
  gather.erase();
  return success();
}

LogicalResult HandleTensorListScatterIntoExistingListOp(
    TF::TensorListScatterIntoExistingListOp scatter,
    toolchain::SmallDenseMap<Value, SizeInfo>* buffer_to_size) {
  auto it = buffer_to_size->find(scatter.getInputHandle());
  if (it == buffer_to_size->end()) {
    return scatter.emitOpError("unknown tensor list");
  }
  auto buffer = scatter.getInputHandle();
  OpBuilder builder(scatter);
  auto indices_type =
      mlir::cast<RankedTensorType>(scatter.getIndices().getType());
  if (!indices_type) return scatter.emitOpError("unranked indices shape");
  auto shape_type = RankedTensorType::get({2}, builder.getIntegerType(32));
  auto shape = TF::ConstOp::create(
      builder, scatter.getLoc(),
      DenseElementsAttr::get(
          shape_type, {static_cast<int>(indices_type.getDimSize(0)), 1}));
  auto indices = TF::ReshapeOp::create(builder, scatter.getLoc(),
                                       scatter.getIndices(), shape);
  Value tensor_scatter_update = TF::TensorScatterUpdateOp::create(
      builder, scatter.getLoc(), buffer, indices, scatter.getTensor());
  scatter.getOutputHandle().replaceAllUsesWith(tensor_scatter_update);
  scatter.erase();
  auto size = it->getSecond();
  (*buffer_to_size)[tensor_scatter_update] = size;
  return success();
}

LogicalResult DecomposeTensorListOpsInternal(
    Block* block, ModuleOp module,
    toolchain::SmallDenseMap<Value, SizeInfo>* buffer_to_size,
    toolchain::StringMap<PartitionedCallDecompositionInfo>*
        decomposed_partitioned_call_callees) {
  for (auto& op : toolchain::make_early_inc_range(block->getOperations())) {
    // TODO(yuanzx): Add a pass to remove identities in device computation.
    if (toolchain::isa<TF::IdentityOp, TF::IdentityNOp, TF::StopGradientOp>(&op)) {
      op.replaceAllUsesWith(op.getOperands());
      op.erase();
    } else if (auto list = toolchain::dyn_cast<TF::EmptyTensorListOp>(&op)) {
      if (failed(HandleEmptyTensorListOp(list, buffer_to_size))) {
        return failure();
      }
    } else if (auto list = toolchain::dyn_cast<TF::TensorListReserveOp>(&op)) {
      if (failed(HandleTensorListReserveOp(list, buffer_to_size))) {
        return failure();
      }
    } else if (auto list = toolchain::dyn_cast<TF::TensorListFromTensorOp>(&op)) {
      if (failed(HandleTensorListFromTensorOp(list, buffer_to_size))) {
        return failure();
      }
    } else if (auto push = toolchain::dyn_cast<TF::TensorListPushBackOp>(&op)) {
      if (failed(HandleTensorListPushBackOp(push, buffer_to_size))) {
        return failure();
      }
    } else if (auto pop = toolchain::dyn_cast<TF::TensorListPopBackOp>(&op)) {
      if (failed(HandleTensorListPopBackOp(pop, buffer_to_size))) {
        return failure();
      }
    } else if (auto get_item = toolchain::dyn_cast<TF::TensorListGetItemOp>(&op)) {
      if (failed(HandleTensorListGetItemOp(get_item, *buffer_to_size))) {
        return failure();
      }
    } else if (auto set_item = toolchain::dyn_cast<TF::TensorListSetItemOp>(&op)) {
      if (failed(HandleTensorListSetItemOp(set_item, buffer_to_size))) {
        return failure();
      }
    } else if (auto length = toolchain::dyn_cast<TF::TensorListLengthOp>(&op)) {
      if (failed(HandleTensorListLengthOp(length, *buffer_to_size))) {
        return failure();
      }
    } else if (auto stack = toolchain::dyn_cast<TF::TensorListStackOp>(&op)) {
      stack.getTensor().replaceAllUsesWith(stack.getInputHandle());
      stack.erase();
    } else if (auto elem_shape =
                   toolchain::dyn_cast<TF::TensorListElementShapeOp>(&op)) {
      if (failed(HandleTensorListElementShapeOp(elem_shape, *buffer_to_size))) {
        return failure();
      }
    } else if (auto gather = toolchain::dyn_cast<TF::TensorListGatherOp>(&op)) {
      if (failed(HandleTensorListGatherOp(gather, *buffer_to_size))) {
        return failure();
      }
    } else if (auto scatter =
                   toolchain::dyn_cast<TF::TensorListScatterIntoExistingListOp>(
                       &op)) {
      if (failed(HandleTensorListScatterIntoExistingListOp(scatter,
                                                           buffer_to_size))) {
        return failure();
      }
    } else if (auto addn = toolchain::dyn_cast<TF::AddNOp>(&op)) {
      auto it = buffer_to_size->find(addn.getOperand(0));
      if (it != buffer_to_size->end()) {
        addn.getSum().setType(
            mlir::cast<TensorType>(addn.getOperand(0).getType()));
        auto size = it->getSecond();
        (*buffer_to_size)[addn.getSum()] = size;
      }
    } else if (auto zeros = toolchain::dyn_cast<TF::ZerosLikeOp>(&op)) {
      if (buffer_to_size->count(zeros.getX()) > 0) {
        zeros.getY().setType(zeros.getX().getType());
        auto size = (*buffer_to_size)[zeros.getX()];
        (*buffer_to_size)[zeros.getY()] = size;
      }
    } else if (auto while_op = toolchain::dyn_cast<TF::WhileOp>(&op)) {
      if (failed(HandleWhileOp(while_op, module, buffer_to_size,
                               decomposed_partitioned_call_callees))) {
        return failure();
      }
    } else if (auto if_op = toolchain::dyn_cast<TF::IfOp>(&op)) {
      if (failed(HandleCaseOrIfOp(
              if_op, {if_op.then_function(), if_op.else_function()}, module,
              buffer_to_size, decomposed_partitioned_call_callees))) {
        return failure();
      }
    } else if (auto case_op = toolchain::dyn_cast<TF::CaseOp>(&op)) {
      SmallVector<func::FuncOp, 2> branches;
      case_op.get_branch_functions(branches);
      if (failed(HandleCaseOrIfOp(case_op, branches, module, buffer_to_size,
                                  decomposed_partitioned_call_callees))) {
        return failure();
      }
    } else if (auto pcall = toolchain::dyn_cast<TF::PartitionedCallOp>(&op)) {
      if (!pcall.func())
        return pcall.emitOpError(
            "TensorList decomposition does not support call with nested "
            "references.");

      if (failed(HandlePartitionedCallOp(
              pcall, pcall.func(), module, buffer_to_size,
              decomposed_partitioned_call_callees))) {
        return failure();
      }
    } else if (auto spcall =
                   toolchain::dyn_cast<TF::StatefulPartitionedCallOp>(&op)) {
      if (failed(HandlePartitionedCallOp(
              spcall, spcall.func(), module, buffer_to_size,
              decomposed_partitioned_call_callees))) {
        return failure();
      }
    } else if (auto while_op = toolchain::dyn_cast<TF::WhileRegionOp>(&op)) {
      if (failed(HandleWhileRegionOp(while_op, module, buffer_to_size,
                                     decomposed_partitioned_call_callees))) {
        return failure();
      }
    } else if (auto if_op = toolchain::dyn_cast<TF::IfRegionOp>(&op)) {
      if (failed(HandleIfRegionOp(if_op, module, buffer_to_size,
                                  decomposed_partitioned_call_callees))) {
        return failure();
      }
    } else if (auto case_op = toolchain::dyn_cast<TF::CaseRegionOp>(&op)) {
      if (failed(HandleCaseRegionOp(case_op, module, buffer_to_size,
                                    decomposed_partitioned_call_callees))) {
        return failure();
      }
    }
  }
  return success();
}

LogicalResult DecomposeTensorListOps(Block* block, ModuleOp module) {
  toolchain::SmallDenseMap<Value, SizeInfo> buffer_to_size;
  toolchain::StringMap<PartitionedCallDecompositionInfo>
      decomposed_partitioned_call_callees;
  return DecomposeTensorListOpsInternal(block, module, &buffer_to_size,
                                        &decomposed_partitioned_call_callees);
}

void TensorListOpsDecompositionPass::runOnOperation() {
  auto module = getOperation();
  auto main = module.lookupSymbol<func::FuncOp>("main");
  if (!main) return;
  if (failed(DecomposeTensorListOps(&main.front(), module))) {
    signalPassFailure();
  }
}

}  // namespace

namespace TF {
std::unique_ptr<OperationPass<ModuleOp>>
CreateTensorListOpsDecompositionPass() {
  return std::make_unique<TensorListOpsDecompositionPass>();
}
}  // namespace TF
}  // namespace mlir
