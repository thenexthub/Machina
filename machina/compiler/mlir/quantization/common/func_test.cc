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
#include "machina/compiler/mlir/quantization/common/func.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/common/test_base.h"

namespace mlir::quant {
namespace {

using ::testing::IsNull;
using ::testing::NotNull;

using FindMainFuncOpTest = ::mlir::quant::QuantizationTestBase;

TEST_F(FindMainFuncOpTest, ReturnsMainFuncOp) {
  constexpr absl::string_view kModuleWithMainFunc = R"mlir(
    module {
      func.func @main() -> () {
        return
      }
    }
  )mlir";

  OwningOpRef<ModuleOp> module_op = ParseModuleOpString(kModuleWithMainFunc);
  EXPECT_THAT(*module_op, NotNull());

  func::FuncOp main_func_op = FindMainFuncOp(*module_op);
  EXPECT_THAT(main_func_op, NotNull());
}

TEST_F(FindMainFuncOpTest, ReturnsNullWhenMainFuncOpIsPrivate) {
  constexpr absl::string_view kModuleWithPrivateMainFunc = R"mlir(
    module {
      func.func private @main() -> () {
        return
      }
    }
  )mlir";

  OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kModuleWithPrivateMainFunc);
  EXPECT_THAT(*module_op, NotNull());

  EXPECT_THAT(FindMainFuncOp(*module_op), IsNull());
}

TEST_F(FindMainFuncOpTest, ReturnsServingDefaultFuncOp) {
  constexpr absl::string_view kModuleWithServingDefaultFunc = R"mlir(
    module {
      func.func @serving_default() -> () {
        return
      }
    }
  )mlir";

  OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kModuleWithServingDefaultFunc);
  EXPECT_THAT(*module_op, NotNull());

  EXPECT_THAT(FindMainFuncOp(*module_op), NotNull());
}

TEST_F(FindMainFuncOpTest, ReturnsNullWhenServingDefaultFuncOpIsPrivate) {
  constexpr absl::string_view kModuleWithPrivateServingDefaultFunc = R"mlir(
    module {
      func.func private @serving_default() -> () {
        return
      }
    }
  )mlir";

  OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kModuleWithPrivateServingDefaultFunc);
  EXPECT_THAT(*module_op, NotNull());

  EXPECT_THAT(FindMainFuncOp(*module_op), IsNull());
}

TEST_F(FindMainFuncOpTest, ReturnsNullWhenMainFuncNotFound) {
  constexpr absl::string_view kModuleWithNoMainFunc = R"mlir(
    module {
      func.func @foo() -> () {
        return
      }
    }
  )mlir";

  OwningOpRef<ModuleOp> module_op = ParseModuleOpString(kModuleWithNoMainFunc);
  EXPECT_THAT(*module_op, NotNull());

  EXPECT_THAT(FindMainFuncOp(*module_op), IsNull());
}

}  // namespace
}  // namespace mlir::quant
