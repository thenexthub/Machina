/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#include "machina/dtensor/mlir/expansions/io_op_spmd_expander.h"

#include <algorithm>
#include <vector>

#include "toolchain/Support/Casting.h"
#include "toolchain/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Block.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/SymbolTable.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/core/platform/errors.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/device_utils.h"
#include "machina/dtensor/mlir/layout_parsing.h"
#include "machina/dtensor/mlir/op_utils.h"
#include "machina/dtensor/mlir/spmd_expander_common.h"
#include "machina/dtensor/mlir/value_utils.h"

namespace machina {
namespace dtensor {

namespace {

template <typename T>
StatusOr<mlir::Operation*> Expand(mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(const std::vector<Layout> output_layouts,
                      ExtractRequiredLayoutFromOp(op));
  TF_ASSIGN_OR_RETURN(const std::vector<Layout> operand_layouts,
                      ExtractRequiredLayoutFromOperands(op));
  if (!AllReplicated(output_layouts) || !AllReplicated(operand_layouts)) {
    return errors::Unimplemented(
        toolchain::formatv("Expecting {0} to have input and output layouts to be "
                      "fully replicated but was not. ",
                      OpName(op))
            .str());
  }

  // Build an if op that only runs the op on device 0. Every other device
  // will run a no-op.
  mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
  mlir::SymbolTable symbol_table(module);
  mlir::Location location = op->getLoc();
  mlir::OpBuilder builder(op);

  auto func_type =
      mlir::FunctionType::get(builder.getContext(), op->getOperandTypes(),
                              toolchain::ArrayRef<mlir::Type>{});
  // Build then_func that is the branch of device_id != 0, which only contains a
  // single NoOp.
  mlir::func::FuncOp then_func = mlir::func::FuncOp::create(
      location,
      toolchain::formatv("{0}_then_func_{1}", OpName(op), OpHash(op)).str(),
      func_type, toolchain::ArrayRef<mlir::NamedAttribute>{});
  // Set function visibility to private to indicate that it is only used in
  // this module.
  then_func.setVisibility(mlir::SymbolTable::Visibility::Private);
  mlir::Block* then_fn_block = then_func.addEntryBlock();
  mlir::OpBuilder then_fn_builder =
      mlir::OpBuilder::atBlockBegin(then_fn_block);
  then_fn_builder.create<mlir::TF::NoOp>(location);
  then_fn_builder.create<mlir::func::ReturnOp>(location);

  // Build else_func that is the branch of device_id == 0.
  // The else func is just the original op.
  mlir::func::FuncOp else_func = mlir::func::FuncOp::create(
      location,
      toolchain::formatv("{0}_else_func_{1}", OpName(op), OpHash(op)).str(),
      func_type, toolchain::ArrayRef<mlir::NamedAttribute>{});
  // Set function visibility to private to indicate that it is only used in
  // this module.
  else_func.setVisibility(mlir::SymbolTable::Visibility::Private);

  mlir::Block* else_fn_block = else_func.addEntryBlock();
  mlir::OpBuilder else_fn_builder =
      mlir::OpBuilder::atBlockBegin(else_fn_block);

  else_fn_builder.create<T>(location, op->getResultTypes(),
                            else_fn_block->getArguments());
  else_fn_builder.create<mlir::func::ReturnOp>(location);

  symbol_table.insert(then_func);
  symbol_table.insert(else_func);

  TF_ASSIGN_OR_RETURN(mlir::Value device_id, DeviceId(op));

  TF_ASSIGN_OR_RETURN(
      mlir::Value zero_scalar,
      CreateZeroScalarConst(
          builder, location,
          mlir::cast<mlir::TensorType>(device_id.getType()).getElementType()));

  mlir::TF::NotEqualOp not_equal = builder.create<mlir::TF::NotEqualOp>(
      location, device_id, zero_scalar,
      /*incompatible_shape_error=*/builder.getBoolAttr(false));

  mlir::Operation* if_op = builder.create<mlir::TF::IfOp>(
      location, then_func.getFunctionType().getResults(),
      /*cond=*/not_equal.getResult(),
      /*input=*/op->getOperands(),
      /*then_branch=*/then_func.getSymName(),
      /*else_branch=*/else_func.getSymName(), /*is_stateless=*/false);

  op->replaceAllUsesWith(if_op);
  op->erase();
  return if_op;
}

}  // namespace
StatusOr<mlir::Operation*> IOOpSPMDExpander::ExpandOp(mlir::Operation* op) {
  if (toolchain::isa<mlir::TF::WriteSummaryOp>(op)) {
    return Expand<mlir::TF::WriteSummaryOp>(op);
  } else if (toolchain::isa<mlir::TF::FlushSummaryWriterOp>(op)) {
    return Expand<mlir::TF::FlushSummaryWriterOp>(op);
  }
  return errors::Unimplemented(
      toolchain::formatv("SPMD for op : {0} is not implemented ", OpName(op)).str());
}

// Always return a set of replicated layouts for now. If there is a case where
// a dtensor user is writing a large tensor that is sharded, then, we can
// support that in the future.
StatusOr<toolchain::DenseMap<int, Layout>> IOOpSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  toolchain::DenseMap<int, Layout> output_layouts(op->getNumResults());
  for (int i = 0; i < op->getNumResults(); ++i) {
    output_layouts[i] =
        Layout::ReplicatedOnMesh(mesh, ValueRank(op->getResult(i)));
  }
  return output_layouts;
}

// Always return a set of replicated layouts. IO ops usually either have
// no output or a scalar output, in which case it is replicated.
StatusOr<toolchain::DenseMap<int, Layout>> IOOpSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  toolchain::DenseMap<int, Layout> input_layouts(op->getNumOperands());
  for (int i = 0; i < op->getNumOperands(); ++i) {
    int rank = std::max(0, ValueRank(op->getOperand(i)));
    input_layouts[i] = Layout::ReplicatedOnMesh(mesh, rank);
  }
  return input_layouts;
}

}  // namespace dtensor
}  // namespace machina
