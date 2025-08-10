/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#include "machina/compiler/mlir/machina/transforms/tf_saved_model_freeze_variables.h"

#include <tuple>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/MapVector.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SetVector.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/analysis/resource_value_typed_analyzer.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_saved_model.h"
#include "machina/compiler/mlir/machina/transforms/tf_saved_model_freeze_utils.h"
#include "machina/compiler/mlir/machina/utils/convert_tensor.h"
#include "machina/compiler/mlir/machina/utils/session_utils.h"
#include "machina/core/framework/resource_var.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/public/session.h"

namespace mlir {
namespace tf_saved_model {
namespace {

// Attribute name that specifies the input shapes of a function.
constexpr StringRef kTfInputShapesAttr = "tf._input_shapes";

// Build and returns ElementsAttr which holds the data in 'tensor'.
ElementsAttr GetTensorValueAsElementsAttr(const machina::Tensor& tensor,
                                          OpBuilder builder) {
  absl::StatusOr<ElementsAttr> tensor_attr_or =
      machina::ConvertTensor(tensor, &builder);
  if (!tensor_attr_or.ok()) return nullptr;
  return tensor_attr_or.value();
}

// Returns ElementsAttr which has the value held by 'resource_tensor'.
ElementsAttr GetTensorValueAsElementsAttr(
    TF::VarHandleOp var_handle_op, const machina::Tensor& resource_tensor,
    const machina::DeviceMgr* mgr, OpBuilder builder) {
  if (resource_tensor.dtype() != machina::DT_RESOURCE) {
    return GetTensorValueAsElementsAttr(resource_tensor, builder);
  }

  auto handle = resource_tensor.scalar<machina::ResourceHandle>()();
  auto* var_ptr = tf_saved_model::GetVariableFromSession(var_handle_op,
                                                         handle.device(), mgr);
  if (!var_ptr) {
    return nullptr;
  }
  machina::core::RefCountPtr<machina::Var> var(var_ptr);
  auto* tensor = var_ptr->tensor();

  return GetTensorValueAsElementsAttr(*tensor, builder);
}

// Returns ID for identifying a resource.
std::tuple<toolchain::StringRef, toolchain::StringRef, toolchain::StringRef> GetResourceKey(
    Operation* op) {
  toolchain::StringRef device;
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("device")) {
    device = attr.getValue();
  }

  toolchain::StringRef container;
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("container")) {
    container = attr.getValue();
  }

  toolchain::StringRef shared_name;
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("shared_name")) {
    shared_name = attr.getValue();
  }

  return std::tuple<toolchain::StringRef, toolchain::StringRef, toolchain::StringRef>{
      device, container, shared_name};
}

// Remove the initialization of the variables in 'var_handle_ops' from
// the session init function 'session_init_func'
void RemoveVariablesInitializations(
    const toolchain::SmallVector<TF::VarHandleOp, 4>& var_handle_ops,
    func::FuncOp session_init_func) {
  // We identify the variables using (device, container, shared_name) of the
  // resource. Capture them here and use them to identify the useless
  // initializations.
  toolchain::SetVector<std::tuple<toolchain::StringRef, toolchain::StringRef, toolchain::StringRef>>
      variables;
  for (auto var_handle_op : var_handle_ops)
    variables.insert(GetResourceKey(var_handle_op));

  toolchain::SmallVector<Operation*, 4> work_list;
  for (auto var_handle_op : session_init_func.getOps<TF::VarHandleOp>()) {
    if (variables.count(GetResourceKey(var_handle_op)))
      work_list.push_back(var_handle_op);
  }

  // Capture list of ops to be erased by traversing usage starting from
  // the VarHandle ops.
  toolchain::SetVector<Operation*> erase_list;
  while (!work_list.empty()) {
    auto* operation = work_list.pop_back_val();
    erase_list.insert(operation);
    for (auto& use : operation->getUses()) {
      if (erase_list.count(use.getOwner())) continue;
      work_list.push_back(use.getOwner());
    }
  }

  for (auto* op : erase_list) {
    op->dropAllUses();
    op->erase();
  }
}

// Validates func ops. Returns `failure` if the function is invalid.
LogicalResult ValidateFuncOp(func::FuncOp func_op) {
  auto input_shapes_attr =
      func_op->getAttrOfType<ArrayAttr>(kTfInputShapesAttr);
  if (!input_shapes_attr) return success();

  if (input_shapes_attr.size() != func_op.getNumArguments()) {
    return func_op->emitError(
               "Number of arguments and 'tf._input_shapes' "
               "attribute size do not match. ")
           << "Num args: " << func_op.getNumArguments()
           << ", tf._input_shapes size: " << input_shapes_attr.size();
  }

  return success();
}

// Validates ModuleOp. Returns `failure` if the module op is invalid.
LogicalResult ValidateModule(ModuleOp module_op) {
  for (auto func_op : module_op.getOps<func::FuncOp>()) {
    if (failed(ValidateFuncOp(func_op))) {
      return failure();
    }
  }
  return success();
}

}  // namespace

LogicalResult FreezeVariables(ModuleOp module, machina::Session* session) {
  if (failed(ValidateModule(module))) return failure();

  const machina::DeviceMgr* mgr = nullptr;
  auto status = session->LocalDeviceManager(&mgr);
  if (!status.ok()) {
    module->emitError(
        absl::StrCat("failed to fetch device manager: ", status.message()));
    return failure();
  }

  SmallVector<func::FuncOp, 2> session_init_funcs =
      tf_saved_model::GetInitializerFunctions(module);
  func::FuncOp session_init_func =
      session_init_funcs.empty() ? nullptr : session_init_funcs[0];

  TF::ResourceAnalyzer analyzer(module, /*skip_session_init=*/true);
  toolchain::SmallVector<TF::VarHandleOp, 4> variables;
  // Capture list of all read only variables.
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func == session_init_func) continue;
    for (auto var_handle_op : func.getOps<TF::VarHandleOp>()) {
      if (!analyzer.IsPotentiallyWritten(var_handle_op.getResource())) {
        variables.push_back(var_handle_op);
      }
    }
  }

  // Fetch the values to replace the VarHandleOps with.
  auto resource_tensors_or =
      tf_saved_model::GetResourcesFromSession(variables, session);
  if (!resource_tensors_or.ok()) {
    module->emitError(resource_tensors_or.status().message().data());
    return failure();
  }

  auto* context = module.getContext();
  OpBuilder builder(context);
  // Note: We can't modify the graph while navigating through it, as erasing
  // invalidate pointers.
  // So instead we capture all the updates in the below map, and then
  // process them after.

  // Container to hold all update actions on ops.
  // Key: Operation to update.
  // Value: optional list of argument indices to delete from this op.
  // Note that we use MapVector because we want to iterate on the same order
  // of insertion.
  toolchain::MapVector<Operation*, toolchain::SmallVector<unsigned int, 4>>
      arguments_to_erase;
  for (auto [var_handle_op, resource_tensor] :
       toolchain::zip(variables, resource_tensors_or.value())) {
    builder.setInsertionPointAfterValue(var_handle_op);
    auto elements_attr = GetTensorValueAsElementsAttr(
        var_handle_op, resource_tensor, mgr, builder);
    if (failed(ReplaceVarWithConstant(var_handle_op.getResource().getUses(),
                                      elements_attr, &arguments_to_erase))) {
      return failure();
    }
  }

  if (failed(EraseObsoleteResourceUses(arguments_to_erase))) {
    return failure();
  }

  // Remove initialization of unused variables.
  if (session_init_func)
    RemoveVariablesInitializations(variables, session_init_func);

  // Remove the unused VarHandleOp.
  for (auto var_handle_op : variables) {
    if (var_handle_op) var_handle_op->erase();
  }
  return success();
}

}  // namespace tf_saved_model
}  // namespace mlir
