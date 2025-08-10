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

#include "machina/compiler/mlir/machina/utils/mlprogram_util.h"

#include "toolchain/ADT/STLFunctionalExtras.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/ADT/Twine.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/transforms/mlprogram.h"

namespace machina {

void RegisterMlProgramPasses() {
  mlir::registerPassPipeline(
      "tf-lower-to-mlprogram-and-hlo", "Lower TF to ml_program + mhlo",
      [](mlir::OpPassManager& pm, toolchain::StringRef options,
         toolchain::function_ref<mlir::LogicalResult(const toolchain::Twine&)>
             errorHandler) {
        machina::PopulateLowerToMlProgramAndHloPipeline(pm);
        return mlir::success();
      },
      [](toolchain::function_ref<void(const mlir::detail::PassOptions&)>) {});
}

}  // namespace machina
