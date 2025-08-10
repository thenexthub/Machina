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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypeInterfaces.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/IR/ValueRange.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Interfaces/FunctionInterfaces.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_ops_n_z.h"
#include "machina/dtensor/cc/constants.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/mlir/value_utils.h"

namespace machina {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORSPARSETENSORTODENSETENSOR
#include "machina/dtensor/mlir/dtensor_passes.h.inc"

constexpr char kEntryFuncAttr[] = "tf.entry_function";
constexpr char kSparseIndicesStr[] = "op_input_sparse_indices";
constexpr char kSparseDenseShapesStr[] = "op_input_sparse_dense_shapes";
constexpr char kSparseValuesStr[] = "op_input_sparse_values";

typedef struct SparseTensorToComponentInfo {
  mlir::RankedTensorType indices;
  mlir::RankedTensorType values;
  mlir::RankedTensorType dense_shapes;
  unsigned int func_op_arg_index;
} SparseTensorToComponentInfo;

void UpdateFunctionSignature(mlir::func::FuncOp function,
                             mlir::OpBuilder& builder) {
  function.setType(mlir::FunctionType::get(
      builder.getContext(),
      toolchain::to_vector<4>(function.front().getArgumentTypes()),
      function.getFunctionType().getResults()));
}

// Add input attributes for new sparsetensor components and remove the
// old sparsetensor value input attributes.
//
// TF has a list of comma separated input names within `kEntryFuncAttr`
// attribute, under 'inputs'. Update this comma separated list of input names
// by correctly deleting the sparse tensor input name and replacing it with
// three new sparse component input names.
//
// Without this update, MLIR conversion to GraphDef will fail since
// the number of input names will not match with the FuncOp num arguments.
//
// e.g. "op_input_1" should become
// "op_input_sparse_indices_0,op_input_sparse_dense_shapes_0,
// "op_input_sparse_values_0"
mlir::LogicalResult UpdateFunctionInputAttributes(
    mlir::MLIRContext& context, mlir::func::FuncOp main_func,
    mlir::OpBuilder& builder,
    const std::vector<SparseTensorToComponentInfo>& sparse_tensor_components) {
  toolchain::SmallVector<toolchain::StringRef, 2> input_names;

  auto dict_attr =
      main_func->getAttrOfType<mlir::DictionaryAttr>(kEntryFuncAttr);
  if (dict_attr) {
    if (!mlir::isa<mlir::StringAttr>(dict_attr.get("inputs")))
      return main_func.emitOpError("Missing attribute inputs in main FuncOp.");

    mlir::cast<mlir::StringAttr>(dict_attr.get("inputs"))
        .getValue()
        .split(input_names, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);

    toolchain::SmallVector<std::string, 2> new_input_names;

    absl::flat_hash_set<int> skip_indices;
    for (const auto component : sparse_tensor_components) {
      skip_indices.insert(component.func_op_arg_index);
    }

    for (auto i = 0; i < input_names.size(); ++i) {
      if (skip_indices.find(i) == skip_indices.end()) {
        new_input_names.push_back(input_names[i].str());
      }
    }

    for (const auto component : sparse_tensor_components) {
      int arg_index = component.func_op_arg_index;
      new_input_names.push_back(
          absl::StrCat(kSparseIndicesStr, "_", arg_index));
      new_input_names.push_back(
          absl::StrCat(kSparseDenseShapesStr, "_", arg_index));
      new_input_names.push_back(absl::StrCat(kSparseValuesStr, "_", arg_index));
    }

    mlir::NamedAttrList attributes(dict_attr);
    attributes.set(
        "inputs",
        mlir::StringAttr::get(&context, absl::StrJoin(new_input_names, ",")));
    main_func->setAttr(kEntryFuncAttr, attributes.getDictionary(&context));
  }
  UpdateFunctionSignature(main_func, builder);
  return mlir::success();
}

// For each SparseTensor block argument of the main FuncOp, create
// three of the component tensors, `indices`, `values`, and `dense_shapes`
// and add it to `sparse_tensor_components`.
void CreateComponentTensorsFromSparseTensors(
    mlir::func::FuncOp main_func, mlir::OpBuilder& builder,
    std::vector<SparseTensorToComponentInfo>* sparse_tensor_components) {
  for (const auto block_arg : main_func.getArguments()) {
    const auto is_sparse = main_func.getArgAttrOfType<mlir::BoolAttr>(
        block_arg.getArgNumber(), kSparseValue);
    if (is_sparse) {
      sparse_tensor_components->push_back(SparseTensorToComponentInfo{
          /*indices=*/mlir::RankedTensorType::get(
              {mlir::ShapedType::kDynamic, ValueRank(block_arg)},
              builder.getI64Type()),
          /*values=*/
          mlir::RankedTensorType::get(
              {mlir::ShapedType::kDynamic},
              mlir::dyn_cast<mlir::RankedTensorType>(block_arg.getType())
                  .getElementType()),
          /*dense_shapes=*/
          mlir::RankedTensorType::get({ValueRank(block_arg)},
                                      builder.getI64Type()),
          /*func_op_arg_index=*/block_arg.getArgNumber()});
    }
  }
}

// Inserts SparseTensor components `components` into `main_func` at the end
// of block arguments list.
void UpdateFunctionWithSparseTensorComponents(
    mlir::MLIRContext& context, mlir::func::FuncOp main_func,
    mlir::OpBuilder& builder, const SparseTensorToComponentInfo& component) {
  main_func.front().addArgument(component.indices, main_func.getLoc());
  main_func.front().addArgument(component.dense_shapes, main_func.getLoc());
  main_func.front().addArgument(component.values, main_func.getLoc());
  UpdateFunctionSignature(main_func, builder);
}

struct DTensorSparseTensorToDenseTensor
    : public impl::DTensorSparseTensorToDenseTensorBase<
          DTensorSparseTensorToDenseTensor> {
  void runOnOperation() override {
    mlir::MLIRContext& context = getContext();
    auto module = getOperation();
    mlir::OpBuilder builder(&context);

    mlir::func::FuncOp main_func =
        module.lookupSymbol<mlir::func::FuncOp>("main");

    // Save Arg Attributes for each argument for later use, this will be
    // reset and reordered after we insert sparse tensor components arguments.
    toolchain::DenseMap<mlir::Value, toolchain::ArrayRef<mlir::NamedAttribute>>
        arg_attribute_map;
    for (auto block_arg : main_func.getArguments()) {
      toolchain::ArrayRef<mlir::NamedAttribute> attrs =
          mlir::function_interface_impl::getArgAttrs(main_func,
                                                     block_arg.getArgNumber());
      arg_attribute_map.insert(std::make_pair(block_arg, attrs));
    }

    std::vector<SparseTensorToComponentInfo> sparse_tensor_components;
    CreateComponentTensorsFromSparseTensors(main_func, builder,
                                            &sparse_tensor_components);

    // Update func arguments in place by replacing SparseTensors with their
    // components and emitting a SparseToDenseOp before all ops that consume
    // a SparseTensor.
    for (const SparseTensorToComponentInfo& components :
         sparse_tensor_components) {
      // Insert SparseTensor component into the main function's block
      // arguments.
      mlir::Value sparse_tensor_value =
          main_func.getArgument(components.func_op_arg_index);

      UpdateFunctionWithSparseTensorComponents(context, main_func, builder,
                                               components);
      mlir::Operation* front_op = &main_func.front().front();
      builder.setInsertionPoint(front_op);

      // Emit a SparseToDenseOp and replace the SparseTensor with the result of
      // this new op.
      StatusOr<mlir::Value> zero_scalar = CreateZeroScalarConst(
          builder, front_op->getLoc(),
          mlir::cast<mlir::TensorType>(sparse_tensor_value.getType())
              .getElementType());
      if (!zero_scalar.ok()) return signalPassFailure();
      mlir::TF::SparseToDenseOp sparse_to_dense_op =
          mlir::TF::SparseToDenseOp::create(
              builder, front_op->getLoc(), sparse_tensor_value.getType(),
              mlir::ValueRange(
                  {main_func.getArgument(main_func.getNumArguments() - 3),
                   main_func.getArgument(main_func.getNumArguments() - 2),
                   main_func.getArgument(main_func.getNumArguments() - 1),
                   zero_scalar.value()}));

      sparse_tensor_value.replaceAllUsesWith(sparse_to_dense_op);
      if (!sparse_tensor_value.use_empty()) return signalPassFailure();
    }

    // Erase sparse tensor arguments now that we converted all of them.
    for (int i = 0; i < sparse_tensor_components.size(); ++i)
      main_func.front().eraseArgument(
          sparse_tensor_components[i].func_op_arg_index - i);

    // Reset block argument attributes since they are likely mixed up
    // due to change in ordering of arguments.
    for (auto block_arg : main_func.getArguments()) {
      if (arg_attribute_map.find(block_arg) == arg_attribute_map.end()) {
        main_func.setArgAttrs(block_arg.getArgNumber(),
                              toolchain::ArrayRef<mlir::NamedAttribute>{});
      } else {
        main_func.setArgAttrs(block_arg.getArgNumber(),
                              arg_attribute_map[block_arg]);
      }
    }
    if (mlir::failed(UpdateFunctionInputAttributes(context, main_func, builder,
                                                   sparse_tensor_components)))
      return signalPassFailure();
  };
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorSparseTensorToDenseTensor() {
  return std::make_unique<DTensorSparseTensorToDenseTensor>();
}

}  // namespace dtensor
}  // namespace machina
