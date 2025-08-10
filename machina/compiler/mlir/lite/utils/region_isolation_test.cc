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

#include "machina/compiler/mlir/lite/utils/region_isolation.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinDialect.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/DialectRegistry.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OpDefinition.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/IR/ValueRange.h"  // part of Codira Toolchain
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo

namespace mlir::TFL {
namespace {

using ::testing::UnorderedElementsAreArray;

static constexpr int64_t kValShape[] = {2, 2};

RankedTensorType TestValType(OpBuilder& b) {
  return RankedTensorType::get(kValShape, b.getI32Type());
}

DenseElementsAttr TestValData(int fill, OpBuilder& b) {
  return b
      .getI32TensorAttr(
          toolchain::SmallVector<int32_t, 4>(TestValType(b).getNumElements(), fill))
      .reshape(TestValType(b));
}

TEST(RegionIsolationTest, CaseOp) {
  //
  // Make Test Model
  //

  //   %0 = const : tensor<2x2xi32>
  //   %1 = const : tensor<2x2xi32>
  //   %2 = const : tensor<2x2xi32>
  //   %3 = const : tensor<i32>
  //   %4 = "stablehlo.case"(%3) ({
  //     %7 = add(%0, %0)
  //   }, {
  //     %6 = add(%0, %1)
  //   }, {
  //     %5 = add(%0, %2)
  //   }

  // Setup context, empty module, etc.

  MLIRContext ctx;

  {
    DialectRegistry reg;
    reg.insert<arith::ArithDialect, BuiltinDialect,
               stablehlo::StablehloDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
  }

  OpBuilder b(&ctx);

  OwningOpRef<ModuleOp> root(b.create<ModuleOp>(b.getUnknownLoc()));

  {
    auto& block = root->getBodyRegion().front();
    b.setInsertionPointToStart(&block);
  }

  // Add values to be referenced within later regions.

  auto root_val_1 =
      arith::ConstantOp::create(b, b.getUnknownLoc(), TestValData(1, b));

  auto root_val_2 =
      arith::ConstantOp::create(b, b.getUnknownLoc(), TestValData(2, b));

  auto root_val_3 =
      arith::ConstantOp::create(b, b.getUnknownLoc(), TestValData(3, b));

  // Iteration convenience.
  toolchain::OwningArrayRef<arith::ConstantOp> root_vals(
      {root_val_1, root_val_2, root_val_3});

  // Make a regioned op with computations that reference defined above vals.

  auto root_ind = arith::ConstantOp::create(
      b, b.getUnknownLoc(),
      DenseIntElementsAttr::get(RankedTensorType::get({}, b.getI32Type()),
                                {0}));

  auto regioned_op = stablehlo::CaseOp::create(
      b, root_val_1.getLoc(), toolchain::SmallVector<Type>({TestValType(b)}),
      root_ind,
      /*branch_count=*/3);

  // Populate each branch with a computation that references the
  // above values.

  for (auto [reg, val] : toolchain::zip(regioned_op.getBranches(), root_vals)) {
    auto& block = reg.emplaceBlock();
    b.setInsertionPointToStart(&block);
    auto res = stablehlo::AddOp::create(b, b.getUnknownLoc(), root_val_1, val);
    toolchain::OwningArrayRef<Value> rets({res});
    stablehlo::ReturnOp::create(b, b.getUnknownLoc(), rets);
  }

  //
  // Isolate Regions in Place
  //

  {
    auto result = IsolateRegions(regioned_op.getOperation(), b);
    ASSERT_TRUE(result.has_value());

    auto& vals_defined_above = result.value();
    EXPECT_THAT(vals_defined_above,
                UnorderedElementsAreArray(toolchain::to_vector(toolchain::map_range(
                    root_vals, [](auto op) { return op.getResult(); }))));

    auto& branch_front = regioned_op.getBranches().front();
    ASSERT_EQ(branch_front.getArgumentTypes().size(),
              vals_defined_above.size());

    for (int i = 0; i < vals_defined_above.size(); ++i) {
      EXPECT_EQ(branch_front.getArgumentTypes()[i],
                vals_defined_above[i].getType());
    }

    for (auto& reg : regioned_op.getBranches()) {
      EXPECT_EQ(reg.getArgumentTypes(), branch_front.getArgumentTypes());
    }
  }
}

}  // namespace

}  // namespace mlir::TFL
