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

#ifndef MACHINA_COMPILER_MLIR_TF2MACHINA_XLAAPI_V2_TF_DIALECT_TO_EXECUTOR_H_
#define MACHINA_COMPILER_MLIR_TF2MACHINA_XLAAPI_V2_TF_DIALECT_TO_EXECUTOR_H_

#include "toolchain/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "machina/core/platform/status.h"

namespace machina {
namespace tf2xla {
namespace v2 {

// Given the input Module op that's in the Tensorflow Dialect, convert the MLIR
// module in place to the Tensorflow Executor Dialect. Returns an OK Status if
// success, otherwise failure with an error message.
// The Tensorflow Executor Dialect is required to export an MLIR module to a
// Tensorflow GraphDef. This API will add control dependencies and verify that
// the conversion was successful.
//
// This also converts the Tensorflow Dialect MLIR into the Tensorflow Executor
// dialect that is suitable to be exported to GraphDef. Graph -> MLIR -> Graph
// is not perfectly round trippable, so this API will attempt to make the module
// exportable and verify some properties of the Tensorflow Executor MLIR that
// are required by Graph Export. It will return an error if it cannot.
//
// Input: A MLIR Module in the Tensorflow Dialect with no
// `tf_device.cluster_func` ops.
// Output: A MLIR module in the Tensorflow Executor Dialect.
ABSL_DEPRECATED("Use machina::tf2xla::v2::ConvertGraphToTfExecutor instead.")
absl::Status ExportFromTensorflowDialectToExecutor(
    mlir::ModuleOp module, toolchain::StringRef module_name = toolchain::StringRef());

}  // namespace v2
}  // namespace tf2xla
}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TF2MACHINA_XLAAPI_V2_TF_DIALECT_TO_EXECUTOR_H_
