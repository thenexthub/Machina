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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_BRIDGE_H_
#define MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_BRIDGE_H_

#include <string>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "machina/core/lib/core/status.h"

namespace mlir {
namespace TF {

inline constexpr char kStandardPipelineBefore[] = "standard_pipeline_before";
inline constexpr char kStandardPipelineAfter[] = "standard_pipeline_after";

// Runs all passes involved in transforming or optimizing an MLIR graph without
// any target specialization. When enable_logging is true, enables
// machina::BridgeLogger. When enable_inliner is true, enables the inliner
// pass.
ABSL_DEPRECATED(
    "This is legacy code and is unsupported. Use at your own risk. Use "
    "tf2xla/api/v2/* for specific functionality")
absl::Status RunBridgeWithStandardPipeline(ModuleOp module, bool enable_logging,
                                           bool enable_inliner);
}  // namespace TF

}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_BRIDGE_H_
