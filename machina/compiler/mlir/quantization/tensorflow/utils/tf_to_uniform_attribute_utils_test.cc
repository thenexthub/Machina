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

#include "machina/compiler/mlir/quantization/machina/utils/tf_to_uniform_attribute_utils.h"

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "toolchain/ADT/StringMap.h"
#include "mlir/IR/AsmState.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Block.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Parser/Parser.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/machina/quantization_options.pb.h"
#include "machina/compiler/mlir/machina/ir/tf_dialect.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

namespace mlir::quant {
namespace {

using QuantMethod = machina::quantization::QuantizationMethod::PresetMethod;

class EmptyPatternRewriter : public mlir::PatternRewriter {
 public:
  explicit EmptyPatternRewriter(const OpBuilder& op_builder)
      : mlir::PatternRewriter(op_builder) {}
  ~EmptyPatternRewriter() override = default;
};

class TfToUniformAttributeUtilsTestPeer {
 public:
  explicit TfToUniformAttributeUtilsTestPeer() = delete;
  explicit TfToUniformAttributeUtilsTestPeer(MLIRContext* ctx)
      : rewriter_(OpBuilder(ctx)) {}

  EmptyPatternRewriter rewriter_;
};

class TfToUniformAttributeUtilsTest : public ::testing::Test {
 protected:
  TfToUniformAttributeUtilsTest() : ctx_() {
    ctx_.loadDialect<TF::TensorFlowDialect>();
  }

  MLIRContext ctx_;
};

TF::UniformQuantizedAddOp ParseUniformQuantizedAddOp(
    const absl::string_view add_op_str, Block& block, MLIRContext& ctx) {
  const LogicalResult parse_result =
      parseSourceString(add_op_str, &block, ParserConfig(&ctx));
  EXPECT_TRUE(succeeded(parse_result));

  auto uq_add_op = dyn_cast_or_null<TF::UniformQuantizedAddOp>(block.back());
  EXPECT_TRUE(uq_add_op);

  return uq_add_op;
}

TF::UniformRequantizeOp ParseUniformRequantizedOp(
    const absl::string_view requant_op_str, Block& block, MLIRContext& ctx) {
  const LogicalResult parse_result =
      parseSourceString(requant_op_str, &block, ParserConfig(&ctx));
  EXPECT_TRUE(succeeded(parse_result));

  auto uq_requant_op = dyn_cast_or_null<TF::UniformRequantizeOp>(block.back());
  EXPECT_TRUE(uq_requant_op);

  return uq_requant_op;
}

TEST_F(TfToUniformAttributeUtilsTest, UniformQuantizedAddOpAttributes) {
  TfToUniformAttributeUtilsTestPeer test_peer(&ctx_);

  constexpr absl::string_view kAddOpExpr =
      R"mlir(
      %0 = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656"> : tensor<1x3x2x2x!tf_type.qint32>} : () -> tensor<1x3x2x2x!tf_type.qint32>
      %1 = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656"> : tensor<2x!tf_type.qint32>} : () -> tensor<2x!tf_type.qint32>
      %2 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
      %3 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %4 = "tf.UniformQuantizedAdd"(%0, %1, %2, %3, %2, %3, %2, %3) {device = "", lhs_quantization_axis = -1 : i64, lhs_quantization_max_val = 127 : i64, lhs_quantization_min_val = -127 : i64, output_quantization_axis = -1 : i64, output_quantization_max_val = 127 : i64, output_quantization_min_val = -127 : i64, rhs_quantization_axis = -1 : i64, rhs_quantization_max_val = 127 : i64, rhs_quantization_min_val = -127 : i64} : (tensor<1x3x2x2x!tf_type.qint32>, tensor<2x!tf_type.qint32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>) -> tensor<1x3x2x2x!tf_type.qint32>
      )mlir";

  Block block{};
  TF::UniformQuantizedAddOp op =
      ParseUniformQuantizedAddOp(kAddOpExpr, block, ctx_);

  toolchain::StringMap<Attribute> identifier_to_attr;
  QuantMethod quantization_method =
      machina::quantization::QuantizationMethod::METHOD_STATIC_RANGE_INT8;
  auto res = FillAttributesForUniformQuantizedAddOp(
      test_peer.rewriter_, op, identifier_to_attr, quantization_method,
      /*enable_per_channel_quantization=*/false);
  ASSERT_TRUE(succeeded(res));
  ASSERT_EQ(2147483647, op.getLhsQuantizationMaxValAttr().getInt());
  ASSERT_EQ(-2147483648, op.getLhsQuantizationMinValAttr().getInt());
  ASSERT_EQ(2147483647, op.getRhsQuantizationMaxValAttr().getInt());
  ASSERT_EQ(-2147483648, op.getRhsQuantizationMinValAttr().getInt());
  ASSERT_EQ(2147483647, op.getOutputQuantizationMaxValAttr().getInt());
  ASSERT_EQ(-2147483648, op.getOutputQuantizationMinValAttr().getInt());
  ASSERT_EQ(-1, op.getLhsQuantizationAxisAttr().getInt());
  ASSERT_EQ(-1, op.getRhsQuantizationAxisAttr().getInt());
  ASSERT_EQ(-1, op.getOutputQuantizationAxisAttr().getInt());
}

TEST_F(TfToUniformAttributeUtilsTest, UniformQuantizedRequantizeOpAttributes) {
  TfToUniformAttributeUtilsTestPeer test_peer(&ctx_);

  constexpr absl::string_view kRequantOpExpr =
      R"mlir(
      %0 = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656"> : tensor<1x3x2x2x!tf_type.qint32>, quantization_axis = 3} : () -> tensor<1x3x2x2x!tf_type.qint32>
      %1 = "tf.Const"() {value = dense<1.0> : tensor<2xf32>} : () -> tensor<2xf32>
      %2 = "tf.Const"() {value = dense<2> : tensor<2xi32>} : () -> tensor<2xi32>
      %3 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
      %4 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %5 = "tf.UniformRequantize"(%0, %1, %2, %3, %4) {device = "", input_quantization_axis = 3 : i64, input_quantization_max_val = 127 : i64, input_quantization_min_val = -127 : i64, output_quantization_axis = -1 : i64, output_quantization_max_val = 127 : i64, output_quantization_min_val = -127 : i64} : (tensor<1x3x2x2x!tf_type.qint32>, tensor<2xf32>, tensor<2xi32>, tensor<f32>, tensor<i32>) -> tensor<1x3x2x2x!tf_type.qint8>
      )mlir";

  Block block{};
  TF::UniformRequantizeOp op =
      ParseUniformRequantizedOp(kRequantOpExpr, block, ctx_);

  toolchain::StringMap<Attribute> identifier_to_attr;
  QuantMethod quantization_method =
      machina::quantization::QuantizationMethod::METHOD_STATIC_RANGE_INT8;
  auto res = FillAttributesForUniformRequantizeOp(
      test_peer.rewriter_, op, identifier_to_attr, quantization_method,
      /*enable_per_channel_quantization=*/true);
  ASSERT_TRUE(succeeded(res));
  ASSERT_EQ(2147483647, op.getInputQuantizationMaxValAttr().getInt());
  ASSERT_EQ(-2147483648, op.getInputQuantizationMinValAttr().getInt());
  ASSERT_EQ(127, op.getOutputQuantizationMaxValAttr().getInt());
  ASSERT_EQ(-128, op.getOutputQuantizationMinValAttr().getInt());
  ASSERT_EQ(3, op.getInputQuantizationAxisAttr().getInt());
  ASSERT_EQ(-1, op.getOutputQuantizationAxisAttr().getInt());
}

TEST_F(TfToUniformAttributeUtilsTest,
       UniformQuantizedRequantizeOpAttributes_OutputPerChannel) {
  TfToUniformAttributeUtilsTestPeer test_peer(&ctx_);

  constexpr absl::string_view kRequantOpExpr =
      R"mlir(
      %0 = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656"> : tensor<1x3x2x2x!tf_type.qint32>, quantization_axis = 3} : () -> tensor<1x3x2x2x!tf_type.qint32>
      %1 = "tf.Const"() {value = dense<1.0> : tensor<2xf32>} : () -> tensor<2xf32>
      %2 = "tf.Const"() {value = dense<2> : tensor<2xi32>} : () -> tensor<2xi32>
      %3 = "tf.Const"() {value = dense<1.0> : tensor<2xf32>} : () -> tensor<2xf32>
      %4 = "tf.Const"() {value = dense<0> : tensor<2xi32>} : () -> tensor<2xi32>
      %5 = "tf.UniformRequantize"(%0, %1, %2, %3, %4) {device = "", input_quantization_axis = 3 : i64, input_quantization_max_val = 127 : i64, input_quantization_min_val = -127 : i64, output_quantization_axis = 1 : i64, output_quantization_max_val = 127 : i64, output_quantization_min_val = -127 : i64} : (tensor<1x3x2x2x!tf_type.qint32>, tensor<2xf32>, tensor<2xi32>, tensor<2xf32>, tensor<2xi32>) -> tensor<1x3x2x2x!tf_type.qint8>
      )mlir";

  Block block{};
  TF::UniformRequantizeOp op =
      ParseUniformRequantizedOp(kRequantOpExpr, block, ctx_);

  toolchain::StringMap<Attribute> identifier_to_attr;
  QuantMethod quantization_method =
      machina::quantization::QuantizationMethod::METHOD_STATIC_RANGE_INT8;
  auto res = FillAttributesForUniformRequantizeOp(
      test_peer.rewriter_, op, identifier_to_attr, quantization_method,
      /*enable_per_channel_quantization=*/true);
  ASSERT_TRUE(succeeded(res));
  ASSERT_EQ(2147483647, op.getInputQuantizationMaxValAttr().getInt());
  ASSERT_EQ(-2147483648, op.getInputQuantizationMinValAttr().getInt());
  ASSERT_EQ(127, op.getOutputQuantizationMaxValAttr().getInt());
  ASSERT_EQ(-128, op.getOutputQuantizationMinValAttr().getInt());
  ASSERT_EQ(3, op.getInputQuantizationAxisAttr().getInt());
  ASSERT_EQ(3, op.getOutputQuantizationAxisAttr().getInt());
}

TEST_F(TfToUniformAttributeUtilsTest,
       UniformQuantizedRequantizeOpAttributes_DisablePerChannelQuantization) {
  TfToUniformAttributeUtilsTestPeer test_peer(&ctx_);

  constexpr absl::string_view kRequantOpExpr =
      R"mlir(
      %0 = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656"> : tensor<1x3x2x2x!tf_type.qint32>, quantization_axis = 3} : () -> tensor<1x3x2x2x!tf_type.qint32>
      %1 = "tf.Const"() {value = dense<1.0> : tensor<2xf32>} : () -> tensor<2xf32>
      %2 = "tf.Const"() {value = dense<2> : tensor<2xi32>} : () -> tensor<2xi32>
      %3 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
      %4 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %5 = "tf.UniformRequantize"(%0, %1, %2, %3, %4) {device = "", input_quantization_axis = 3 : i64, input_quantization_max_val = 127 : i64, input_quantization_min_val = -127 : i64, output_quantization_axis = -1 : i64, output_quantization_max_val = 127 : i64, output_quantization_min_val = -127 : i64} : (tensor<1x3x2x2x!tf_type.qint32>, tensor<2xf32>, tensor<2xi32>, tensor<f32>, tensor<i32>) -> tensor<1x3x2x2x!tf_type.qint8>
      )mlir";

  Block block{};
  TF::UniformRequantizeOp op =
      ParseUniformRequantizedOp(kRequantOpExpr, block, ctx_);

  toolchain::StringMap<Attribute> identifier_to_attr;
  QuantMethod quantization_method =
      machina::quantization::QuantizationMethod::METHOD_STATIC_RANGE_INT8;
  auto res = FillAttributesForUniformRequantizeOp(
      test_peer.rewriter_, op, identifier_to_attr, quantization_method,
      /*enable_per_channel_quantization=*/false);
  ASSERT_TRUE(succeeded(res));
  ASSERT_EQ(2147483647, op.getInputQuantizationMaxValAttr().getInt());
  ASSERT_EQ(-2147483648, op.getInputQuantizationMinValAttr().getInt());
  ASSERT_EQ(127, op.getOutputQuantizationMaxValAttr().getInt());
  ASSERT_EQ(-128, op.getOutputQuantizationMinValAttr().getInt());
  ASSERT_EQ(-1, op.getInputQuantizationAxisAttr().getInt());
  ASSERT_EQ(-1, op.getOutputQuantizationAxisAttr().getInt());
}

}  // namespace
}  // namespace mlir::quant
