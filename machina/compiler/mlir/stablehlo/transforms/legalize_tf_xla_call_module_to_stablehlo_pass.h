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
// LINT.IfChange
#ifndef MACHINA_COMPILER_MLIR_STABLEHLO_TRANSFORMS_LEGALIZE_TF_MACHINA_XLACALL_MODULE_TO_STABLEHLO_PASS_H_
#define MACHINA_COMPILER_MLIR_STABLEHLO_TRANSFORMS_LEGALIZE_TF_MACHINA_XLACALL_MODULE_TO_STABLEHLO_PASS_H_

#include <memory>

#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain

namespace mlir {
namespace stablehlo {

// Adds passes which transform TF_XlaCallModule Op to StableHLO Ops.
// Note that this pass only supports static shape tensors for now.
std::unique_ptr<mlir::OperationPass<ModuleOp>>
CreateLegalizeTFXlaCallModuleToStablehloPass();

}  // namespace stablehlo
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_STABLEHLO_TRANSFORMS_LEGALIZE_TF_MACHINA_XLACALL_MODULE_TO_STABLEHLO_PASS_H_
// LINT.ThenChange(//machina/compiler/mlir/lite/stablehlo/transforms/legalize_tf_xla_call_module_to_stablehlo_pass.h)
