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

#include <cassert>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/SmallSet.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/ADT/StringSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/SymbolTable.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops_a_m.h"
#include "machina/compiler/mlir/machina/ir/tf_ops_n_z.h"
#include "machina/compiler/mlir/machina/ir/tf_saved_model.h"
#include "machina/compiler/mlir/machina/utils/convert_tensor.h"
#include "machina/compiler/mlir/machina/utils/session_utils.h"
#include "machina/core/framework/resource_var.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/public/session.h"

namespace mlir {
namespace tf_saved_model {
namespace {

void InitializeVariable(TF::VarHandleOp var_handle_op,
                        machina::Tensor* tensor,
                        func::FuncOp session_init_func, OpBuilder builder) {
  absl::StatusOr<ElementsAttr> tensor_attr_or =
      machina::ConvertTensor(*tensor, &builder);
  assert(tensor_attr_or.ok() && "Expect valid tensor");
  ElementsAttr tensor_attr = tensor_attr_or.value();

  builder.setInsertionPointToStart(&session_init_func.getBlocks().front());
  auto var_handle_op_in_init = var_handle_op->clone();
  builder.insert(var_handle_op_in_init);
  auto const_op = builder.create<mlir::arith::ConstantOp>(
      session_init_func.getLoc(), tensor_attr.getType(), tensor_attr);

  builder.create<TF::AssignVariableOp>(
      session_init_func.getLoc(), toolchain::ArrayRef<mlir::Type>{},
      toolchain::ArrayRef<mlir::Value>{var_handle_op_in_init->getResult(0),
                                  const_op.getResult()});
}

func::FuncOp CreateSessionInitFunc(ModuleOp module) {
  constexpr char kSessionInitFuncName[] = "SessionInitializerFunction";

  mlir::OpBuilder builder(module.getBodyRegion());
  auto func_type =
      FunctionType::get(module.getContext(), /*inputs=*/{}, /*results=*/{});
  auto func = builder.create<func::FuncOp>(module->getLoc(),
                                           kSessionInitFuncName, func_type);
  func->setAttr(kTfSavedModelExportedNamesAttr,
                builder.getStrArrayAttr({kSessionInitFuncName}));
  func->setAttr(kTfSavedModelInitializerTypeAttr,
                builder.getStringAttr(kTfSavedModelInitializerRestoreType));
  func.setVisibility(mlir::func::FuncOp::Visibility::Public);
  auto func_builder = OpBuilder::atBlockBegin(func.addEntryBlock());
  func_builder.create<mlir::func::ReturnOp>(func.getLoc());
  // In cases where there is a session initializer op with empty initializer,
  // replace the session initializer with the new one that points to the session
  // initializer func.
  SessionInitializerOp session_init_op = GetSessionInitializerOp(module);
  auto new_session_init_op =
      builder.create<tf_saved_model::SessionInitializerOp>(
          module->getLoc(), builder.getArrayAttr(SymbolRefAttr::get(
                                builder.getContext(), kSessionInitFuncName)));
  if (session_init_op) {
    session_init_op->replaceAllUsesWith(new_session_init_op);
    session_init_op->erase();
  }
  return func;
}

func::FuncOp GetOrCreateSessionInitFunc(ModuleOp module) {
  SessionInitializerOp session_init_op = GetSessionInitializerOp(module);
  if (!session_init_op) return CreateSessionInitFunc(module);

  auto init_func_op = GetInitializerFunction(
      module, /*initializer_type=*/kTfSavedModelInitializerRestoreType);
  if (init_func_op) {
    return init_func_op;
  } else if (!session_init_op.getInitializers().empty()) {
    // When the init function with type "restore_op" is not found, fall back to
    // taking the init function corresponding to the first symbol in the
    // initializers list to be backwards-compatible, before
    // tf_saved_model.initializer_type attribute was introduced.
    SymbolTable symbol_table(module);
    return symbol_table.lookup<func::FuncOp>(
        mlir::cast<FlatSymbolRefAttr>(session_init_op.getInitializers()[0])
            .getValue());
  } else {
    return CreateSessionInitFunc(module);
  }
}

}  // namespace

LogicalResult InitializeVariablesInSessionInitializer(
    ModuleOp module, machina::Session* session) {
  const machina::DeviceMgr* mgr = nullptr;
  auto status = session->LocalDeviceManager(&mgr);
  if (!status.ok()) {
    module->emitError(
        absl::StrCat("failed to fetch device manager: ", status.message()));
    return failure();
  }

  // Fetch all VarHandleOp.
  toolchain::StringSet<> variable_names;
  toolchain::SmallVector<TF::VarHandleOp, 4> var_ops;
  for (auto func_op : module.getOps<func::FuncOp>()) {
    for (auto var_handle_op : func_op.getOps<TF::VarHandleOp>()) {
      auto variable_name = GetVariableName(var_handle_op);
      if (variable_names.count(variable_name)) continue;
      var_ops.emplace_back(var_handle_op);
      variable_names.insert(variable_name);
    }
  }

  // Get resources from Session.
  auto resource_tensors_or = GetResourcesFromSession(var_ops, session);
  if (!resource_tensors_or.ok()) {
    module->emitError(resource_tensors_or.status().message().data());
    return failure();
  }

  auto session_init_func = GetOrCreateSessionInitFunc(module);
  OpBuilder builder(session_init_func.getContext());

  for (auto var_and_tensor : toolchain::zip(var_ops, resource_tensors_or.value())) {
    auto& var_op = std::get<0>(var_and_tensor);
    auto& resource_tensor = std::get<1>(var_and_tensor);
    if (resource_tensor.dtype() != machina::DT_RESOURCE) {
      InitializeVariable(var_op, &resource_tensor, session_init_func, builder);
      continue;
    }

    auto handle = resource_tensor.scalar<machina::ResourceHandle>()();
    auto* var_ptr = GetVariableFromSession(var_op, handle.device(), mgr);
    if (!var_ptr) {
      // If no value in session, then just skip this variable.
      // This can happen if the variable is not saved in checkpoint.
      // For example, when the variable is created on every call.
      continue;
    }
    machina::core::RefCountPtr<machina::Var> var(var_ptr);
    auto* tensor = var_ptr->tensor();

    InitializeVariable(var_op, tensor, session_init_func, builder);
  }
  return success();
}

}  // namespace tf_saved_model
}  // namespace mlir
