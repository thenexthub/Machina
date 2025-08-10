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
#ifndef MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_PYTHON_UNFREEZE_CONSTANTS_H_
#define MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_PYTHON_UNFREEZE_CONSTANTS_H_

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain

namespace machina {
namespace quantization {

inline constexpr absl::string_view kTfQuantConstantUnfreezingStepName =
    "tf_quant_constant_unfreezing";
inline constexpr absl::string_view kTfQuantInsertRestoreOpStepName =
    "tf_quant_insert_restore_op";

absl::Status UnfreezeConstantsAndSaveVariables(absl::string_view checkpoint_dir,
                                               mlir::MLIRContext &ctx,
                                               mlir::ModuleOp module_op);

}  // namespace quantization
}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_PYTHON_UNFREEZE_CONSTANTS_H_
