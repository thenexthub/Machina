/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_UTILS_EVAL_UTIL_H_
#define MACHINA_COMPILER_MLIR_MACHINA_UTILS_EVAL_UTIL_H_

#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "machina/c/eager/c_api.h"

namespace machina {

// Attempts to evaluates an MLIR Operation in TensorFlow eager mode with the
// specified operands. The op is always executed on the local host CPU
// irrespective of the device attribute of the given op. If there is a CPU
// kernel registered for the op and is executed successfully, this fills in the
// results vector.  If not, results vector is unspecified.
//
mlir::LogicalResult EvaluateOperation(
    mlir::Operation* inst, toolchain::ArrayRef<mlir::ElementsAttr> operands,
    TFE_Context* context, toolchain::SmallVectorImpl<mlir::Attribute>* results);

}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_MACHINA_UTILS_EVAL_UTIL_H_
