/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_EXTRACT_CALLBACK_H_
#define MACHINA_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_EXTRACT_CALLBACK_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "machina/core/framework/types.pb.h"

namespace machina {
namespace ifrt_serving {

// Extracts a module that consists of a public callback function in name of
// `callback_key` and all its reachables.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ExtractCallbackModule(
    mlir::ModuleOp module, absl::string_view callback_key);

}  // namespace ifrt_serving
}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_EXTRACT_CALLBACK_H_
