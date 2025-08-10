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

#ifndef MACHINA_COMPILER_MLIR_LITE_DEBUG_DEBUG_H_
#define MACHINA_COMPILER_MLIR_LITE_DEBUG_DEBUG_H_

#include "toolchain/Support/raw_ostream.h"
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/debug/debug_options.pb.h"

namespace machina {

// Initializes the pass manager with default options that make debugging easier.
// The `out` method parameter is exposed for testing purposes and not intended
// to be specified by client code.
void InitPassManager(mlir::PassManager& pm,
                     const converter::DebugOptions& options,
                     toolchain::raw_ostream& out = toolchain::outs());

}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_LITE_DEBUG_DEBUG_H_
