/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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
#include "machina/compiler/mlir/quantization/machina/python/unfreeze_constants.h"

#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/machina/cc/run_passes.h"
#include "machina/compiler/mlir/quantization/machina/cc/save_variables.h"
#include "machina/compiler/mlir/quantization/machina/passes/passes.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/core/platform/env.h"

namespace machina {
namespace quantization {

// Unfreezes constants into variables and saves them to a checkpoint files under
// `checkpoint_dir`. `checkpoint_dir` will be created within this function. It
// will return a non-OK status if it already exists or permission is denied.
// TODO(b/261652258): Make sure this works for when there are non-frozen
// variables in the model.
absl::Status UnfreezeConstantsAndSaveVariables(
    const absl::string_view checkpoint_dir, mlir::MLIRContext &ctx,
    mlir::ModuleOp module_op) {
  TF_RETURN_IF_ERROR(RunPasses(
      /*name=*/kTfQuantConstantUnfreezingStepName, /*add_passes_func=*/
      [](mlir::PassManager &pm) {
        pm.addPass(mlir::quant::CreateUnfreezeConstantsPass());
      },
      ctx, module_op));

  if (const absl::Status create_dir_status =
          Env::Default()->CreateDir(std::string(checkpoint_dir));
      !create_dir_status.ok()) {
    LOG(ERROR) << "Failed to create checkpoint directory at: "
               << checkpoint_dir;
    return create_dir_status;
  }

  TF_ASSIGN_OR_RETURN(const auto unused_variable_names,
                      SaveVariablesToCheckpoint(checkpoint_dir, module_op));

  return RunPasses(
      /*name=*/kTfQuantInsertRestoreOpStepName,
      /*add_passes_func=*/
      [](mlir::PassManager &pm) {
        pm.addPass(mlir::quant::CreateInsertRestoreOpPass());
        pm.addPass(mlir::quant::CreateInsertSaveOpPass());
        // Initialization by `tf.ConstOp` is no longer required as there is
        // a `tf.RestoreV2Op` now.
        pm.addPass(
            mlir::quant::CreateRemoveVariableInitializationByConstPass());
      },
      ctx, module_op);
}
}  // namespace quantization
}  // namespace machina
