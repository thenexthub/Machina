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

#include "machina/compiler/mlir/tf2xla/transforms/test_utils.h"

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/register_common_dialects.h"
#include "machina/compiler/mlir/machina/dialect_registration.h"
#include "machina/compiler/mlir/machina/utils/deserialize_mlir_module_utils.h"
#include "machina/xla/tsl/platform/statusor.h"

namespace mlir {
namespace hlo {
namespace test {

using ::mlir::DialectRegistry;
using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::OwningOpRef;
using ::tsl::StatusOr;

absl::StatusOr<OwningOpRef<ModuleOp>> GetMlirModuleFromString(
    absl::string_view module_string, MLIRContext* context) {
  DialectRegistry mlir_registry;
  RegisterCommonToolingDialects(mlir_registry);
  context->appendDialectRegistry(mlir_registry);

  OwningOpRef<ModuleOp> mlir_module;
  auto status =
      machina::DeserializeMlirModule(module_string, context, &mlir_module);
  if (!status.ok()) {
    return status;
  }
  return mlir_module;
}

}  // namespace test
}  // namespace hlo
}  // namespace mlir
