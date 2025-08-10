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

#ifndef MACHINA_COMPILER_MLIR_TF2MACHINA_XLAINTERNAL_LOGGING_HOOKS_H_
#define MACHINA_COMPILER_MLIR_TF2MACHINA_XLAINTERNAL_LOGGING_HOOKS_H_

#include <string>

#include "toolchain/ADT/StringRef.h"
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain

namespace machina {
namespace tf2xla {
namespace internal {

// Setup the input pass manager to enable IR dumping after each pass.
// Note a side effect of this method is that multi threading will be disabled.
void EnablePassIRPrinting(mlir::PassManager& pm,
                          const std::string& dump_group_name,
                          toolchain::StringRef module_name = toolchain::StringRef());

};  // namespace internal
};  // namespace tf2xla
};  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TF2MACHINA_XLAINTERNAL_LOGGING_HOOKS_H_
