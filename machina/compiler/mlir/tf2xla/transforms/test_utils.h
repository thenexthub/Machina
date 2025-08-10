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

#ifndef MACHINA_COMPILER_MLIR_TF2MACHINA_XLATRANSFORMS_TEST_UTILS_H_
#define MACHINA_COMPILER_MLIR_TF2MACHINA_XLATRANSFORMS_TEST_UTILS_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/utils/deserialize_mlir_module_utils.h"
#include "machina/xla/tsl/platform/statusor.h"

namespace mlir {
namespace hlo {
namespace test {

// Given a raw string, return a ModuleOp that can be used with the given
// MLIRContext.
absl::StatusOr<OwningOpRef<ModuleOp>> GetMlirModuleFromString(
    absl::string_view module_string, MLIRContext* mlir_context);

}  // namespace test
}  // namespace hlo
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_TF2MACHINA_XLATRANSFORMS_TEST_UTILS_H_
