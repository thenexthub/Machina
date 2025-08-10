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
#include "machina/compiler/mlir/machina/utils/session_utils.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "toolchain/ADT/SmallSet.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/utils/string_container_utils.h"
#include "machina/core/common_runtime/device_mgr.h"
#include "machina/core/framework/device.h"
#include "machina/core/framework/resource_var.h"

namespace mlir {
namespace tf_saved_model {

std::string GetVariableName(TF::VarHandleOp var_handle_op) {
  // In some cases the shared_name attribute doesn't have the same
  // tensor name in the model, so we first try to use the location
  // then fallback to shared_name attribute.
  if (auto loc = mlir::dyn_cast<NameLoc>(var_handle_op->getLoc()))
    return loc.getName().str();
  return var_handle_op.getSharedName().str();
}

std::vector<std::string> GetVariableNames(
    toolchain::ArrayRef<TF::VarHandleOp> var_handle_ops) {
  std::vector<std::string> names;
  names.reserve(var_handle_ops.size());
  for (auto var_handle_op : var_handle_ops)
    names.push_back(GetVariableName(var_handle_op));
  return names;
}

machina::Var* GetVariableFromSession(mlir::TF::VarHandleOp var_handle_op,
                                        toolchain::StringRef device_name,
                                        const machina::DeviceMgr* mgr) {
  machina::Device* device = nullptr;
  if (!mgr || !mgr->LookupDevice(StringRefToView(device_name), &device).ok())
    return nullptr;
  machina::Var* var_ptr = nullptr;
  const auto& container = var_handle_op.getContainer().str();
  auto status = device->resource_manager()->Lookup(
      (container.empty() ? device->resource_manager()->default_container()
                         : container),
      var_handle_op.getSharedName().str(), &var_ptr);
  if (!device || !status.ok()) return nullptr;
  return var_ptr;
}

absl::StatusOr<std::vector<machina::Tensor>> GetResourcesFromSession(
    toolchain::ArrayRef<TF::VarHandleOp> var_handle_ops,
    machina::Session* session) {
  if (!session)
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "Null Session provided.");
  std::vector<machina::Tensor> resource_tensors;
  auto variable_names = GetVariableNames(var_handle_ops);
  if (variable_names.empty()) return resource_tensors;

  auto status = session->Run({}, variable_names, {}, &resource_tensors);
  if (!status.ok())
    return absl::Status(absl::StatusCode::kInternal, status.message());
  return resource_tensors;
}
}  // namespace tf_saved_model
}  // namespace mlir
