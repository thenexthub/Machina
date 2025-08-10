/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#include "machina/compiler/mlir/quantization/machina/utils/tf_to_xla_attribute_utils.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/Support/Debug.h"
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypeInterfaces.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/common/attrs_and_constraints.h"
#include "machina/compiler/mlir/machina/ir/tf_dialect.h"

namespace mlir::quant {
namespace {

void PackOperandTestHelper(
    const toolchain::SmallVector<int64_t>& unpacked_shape,
    const toolchain::SmallVector<int8_t>& unpacked_values, int pack_dim,
    const toolchain::SmallVector<int64_t>& expected_packed_shape,
    const toolchain::SmallVector<int8_t>& expected_packed_values) {
  MLIRContext context;
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
  OpBuilder builder(&module->getBodyRegion());
  context.loadDialect<TF::TensorFlowDialect>();

  Value value = CreateConstValue<int8_t>(builder, module->getLoc(),
                                         unpacked_shape, unpacked_values);
  Value packed_value = PackOperand(builder, module->getLoc(), value, pack_dim);
  DenseIntElementsAttr packed_value_attr;
  ASSERT_TRUE(matchPattern(packed_value, m_Constant(&packed_value_attr)));

  ShapedType packed_shape_type =
      mlir::dyn_cast<ShapedType>(packed_value.getType());
  toolchain::SmallVector<int64_t> packed_shape(packed_shape_type.getShape().begin(),
                                          packed_shape_type.getShape().end());
  EXPECT_THAT(packed_shape, testing::ElementsAreArray(expected_packed_shape));
  toolchain::SmallVector<int8_t> packed_value_vector(
      packed_value_attr.getValues<int8_t>());
  EXPECT_THAT(packed_value_vector,
              testing::ElementsAreArray(expected_packed_values));
}

TEST(TfToXlaAttributeUtilsTest, PackOperandPackDimSizeEven) {
  PackOperandTestHelper(/*unpacked_shape=*/{2, 2},
                        /*unpacked_values=*/{0x01, 0x02, 0x03, 0x04},
                        /*pack_dim=*/0,
                        /*expected_packed_shape=*/{1, 2},
                        /*expected_packed_values=*/{0x31, 0x42});
}

TEST(TfToXlaAttributeUtilsTest, PackOperandPackDimSizeOdd) {
  PackOperandTestHelper(
      /*unpacked_shape=*/{2, 3},
      /*unpacked_values=*/{0x01, 0x02, 0x03, 0x04, 0x05, 0x06},
      /*pack_dim=*/1,
      /*expected_packed_shape=*/{2, 2},
      /*expected_packed_values=*/{0x31, 0x02, 0x64, 0x05});
}

}  // namespace
}  // namespace mlir::quant
