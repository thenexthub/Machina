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

#include "machina/compiler/mlir/op_or_arg_name_mapper.h"

#include <optional>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain

namespace machina {
namespace {

TEST(OpOrArgNameMapperTest, GetMappedNameView) {
  mlir::MLIRContext context;
  OpOrArgLocNameMapper mapper;

  // Create a dummy operation.
  context.allowUnregisteredDialects();
  mlir::OperationState state(mlir::UnknownLoc::get(&context), "test.op");
  mlir::Operation *op = mlir::Operation::create(state);

  // Test case 1: Name not mapped yet.
  EXPECT_EQ(mapper.GetMappedNameView(op), std::nullopt);

  // Map a name.
  mapper.InitOpName(op, "test_op");

  // Test case 2: Name is mapped.
  std::optional<absl::string_view> name = mapper.GetMappedNameView(op);
  EXPECT_TRUE(name.has_value());
  EXPECT_EQ(*name, "test_op");

  // Clean up the operation.
  op->destroy();
}

}  // namespace
}  // namespace machina
