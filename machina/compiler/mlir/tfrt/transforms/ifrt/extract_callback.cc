/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#include "machina/compiler/mlir/tfrt/transforms/ifrt/extract_callback.h"

#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops_a_m.h.inc"
#include "machina/compiler/mlir/machina/ir/tf_ops_n_z.h.inc"
#include "machina/compiler/mlir/machina/utils/error_util.h"
#include "machina/compiler/mlir/machina/utils/visitor.h"
#include "machina/core/framework/tensor_shape.pb.h"
#include "machina/core/framework/types.pb.h"

namespace machina {
namespace ifrt_serving {

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ExtractCallbackModule(
    mlir::ModuleOp module, absl::string_view callback_key) {
  // Find the entry function name first.
  mlir::func::FuncOp callback_entry_func;
  module.walk([&](mlir::func::FuncOp func) {
    if (func.getSymName().str() == callback_key) {
      callback_entry_func = func;
      return mlir::WalkResult::skip();
    }
    return mlir::WalkResult::advance();
  });

  if (!callback_entry_func) {
    return absl::NotFoundError(
        absl::StrCat("Callback key ", callback_key, " not found"));
  }

  mlir::StatusScopedDiagnosticHandler diag_handler(module->getContext());
  auto entry_function_name = callback_entry_func.getSymName();
  auto submodule = mlir::TF::CreatePrunedModule(module, entry_function_name);
  if (mlir::failed(submodule)) {
    return diag_handler.ConsumeStatus();
  }

  // Remove the attribute inherited from saved model loading. They impose
  // additional constraint on public functions that are not necessary.
  submodule->get()->removeAttr("tf_saved_model.semantics");
  submodule->get().walk([&](mlir::func::FuncOp func) {
    if (func.getSymName() == entry_function_name) {
      func.setPublic();
    }
  });
  return std::move(*submodule);
}

}  // namespace ifrt_serving
}  // namespace machina
