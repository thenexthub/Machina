/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, August 10, 2025.
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
#include "machina/compiler/mlir/machina/utils/serialize_mlir_module_utils.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "machina/compiler/jit/flags.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace {

TEST(SerializeMlirModuleUtilsTest, DebugInfoSerialization) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

  GetMlirCommonFlags()->tf_mlir_enable_debug_info_serialization = true;
  std::string serialized_module = SerializeMlirModule(*mlir_module);
  EXPECT_TRUE(absl::StrContains(serialized_module, "loc("));

  GetMlirCommonFlags()->tf_mlir_enable_debug_info_serialization = false;
  serialized_module = SerializeMlirModule(*mlir_module);
  EXPECT_FALSE(absl::StrContains(serialized_module, "loc("));
}

}  // namespace
}  // namespace machina
