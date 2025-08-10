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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_UTILS_STABLEHLO_CUSTOM_CALL_H_
#define MACHINA_COMPILER_MLIR_MACHINA_UTILS_STABLEHLO_CUSTOM_CALL_H_

#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo

namespace mlir {
namespace TF {

// Returns whether the custom call op represents a TF function call.
bool IsTfFuncCustomCall(stablehlo::CustomCallOp op);

// Returns the `called_func` symbol ref attribute in the `tf.backend_config`
// dictionary attribute.
//
// If the op does not represent a TF function call, returns nullptr.
// Otherwise, if the op does not have `caller_name`, returns failure.
FailureOr<SymbolRefAttr> GetTfFuncCustomCallFuncName(
    stablehlo::CustomCallOp op);

}  // namespace TF
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_MACHINA_UTILS_STABLEHLO_CUSTOM_CALL_H_
