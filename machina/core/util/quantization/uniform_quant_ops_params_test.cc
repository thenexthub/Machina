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
#include "machina/core/util/quantization/uniform_quant_ops_params.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/xla/tsl/lib/core/status_test_util.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/util/quantization/uniform_quant_ops_attr.pb.h"

namespace machina {
namespace {

using protobuf::TextFormat;
using ::testing::ElementsAreArray;

TEST(UniformQuantizedConvolutionParamsTest, DilatedSize) {
  EXPECT_EQ(UniformQuantizedConvolutionParams::DilatedSize(0, 2), 0);
  EXPECT_EQ(UniformQuantizedConvolutionParams::DilatedSize(10, 3), 28);
}

TEST(UniformQuantizedConvolutionParamsTest,
     ValidateOrFillParamsAndValidateShapeDefaultAttr) {
  UniformQuantizedConvolutionDimensionNumbersAttr dimension_numbers;
  UniformQuantizedConvolutionParams params(/*window_strides=*/{},
                                           /*lhs_dilation=*/{},
                                           /*rhs_dilation=*/{},
                                           dimension_numbers,
                                           /*feature_group_count=*/1,
                                           /*batch_group_count=*/1,
                                           /*padding=*/"VALID");
  TF_ASSERT_OK(
      params.ValidateOrFillParamsAndValidateShape(/*lhs_shape=*/{2, 2, 3, 4},
                                                  /*rhs_shape=*/{3, 2, 2, 3}));

  EXPECT_THAT(params.window_strides(), ElementsAreArray({1, 1}));
  EXPECT_THAT(params.lhs_dilation(), ElementsAreArray({1, 1}));
  EXPECT_THAT(params.rhs_dilation(), ElementsAreArray({1, 1}));
  EXPECT_THAT(params.padding_list(), ElementsAreArray({0, 0, 0, 0}));
  EXPECT_EQ(params.dimension_numbers().input_batch_dimension(), 0);
  EXPECT_EQ(params.dimension_numbers().input_feature_dimension(), 1);
  EXPECT_THAT(params.dimension_numbers().input_spatial_dimensions(),
              ElementsAreArray({2, 3}));
  EXPECT_EQ(params.dimension_numbers().kernel_output_feature_dimension(), 0);
  EXPECT_EQ(params.dimension_numbers().kernel_input_feature_dimension(), 1);
  EXPECT_THAT(params.dimension_numbers().kernel_spatial_dimensions(),
              ElementsAreArray({2, 3}));
  EXPECT_EQ(params.dimension_numbers().output_batch_dimension(), 0);
  EXPECT_EQ(params.dimension_numbers().output_feature_dimension(), 1);
  EXPECT_THAT(params.dimension_numbers().output_spatial_dimensions(),
              ElementsAreArray({2, 3}));
}

TEST(UniformQuantizedConvolutionParamsTest,
     ValidateOrFillParamsAndValidateShapeSetAttr) {
  UniformQuantizedConvolutionDimensionNumbersAttr dimension_numbers;
  ASSERT_TRUE(TextFormat::ParseFromString(R"pb(
                                            input_batch_dimension: 0
                                            input_feature_dimension: 3
                                            input_spatial_dimensions: 1
                                            input_spatial_dimensions: 2
                                            kernel_output_feature_dimension: 3
                                            kernel_input_feature_dimension: 2
                                            kernel_spatial_dimensions: 0
                                            kernel_spatial_dimensions: 1
                                            output_batch_dimension: 0
                                            output_feature_dimension: 3
                                            output_spatial_dimensions: 1
                                            output_spatial_dimensions: 2
                                          )pb",
                                          &dimension_numbers));
  UniformQuantizedConvolutionParams params(/*window_strides=*/{2, 2},
                                           /*lhs_dilation=*/{3, 3},
                                           /*rhs_dilation=*/{4, 4},
                                           dimension_numbers,
                                           /*feature_group_count=*/2,
                                           /*batch_group_count=*/1,
                                           /*padding=*/"EXPLICIT",
                                           /*padding_list=*/{1, 1, 2, 2});
  TF_ASSERT_OK(
      params.ValidateOrFillParamsAndValidateShape(/*lhs_shape=*/{2, 3, 4, 2},
                                                  /*rhs_shape=*/{2, 3, 1, 2}));

  EXPECT_THAT(params.padding_list(), ElementsAreArray({1, 1, 2, 2}));
  EXPECT_EQ(params.dimension_numbers().input_batch_dimension(), 0);
  EXPECT_EQ(params.dimension_numbers().input_feature_dimension(), 3);
  EXPECT_THAT(params.dimension_numbers().input_spatial_dimensions(),
              ElementsAreArray({1, 2}));
  EXPECT_EQ(params.dimension_numbers().kernel_output_feature_dimension(), 3);
  EXPECT_EQ(params.dimension_numbers().kernel_input_feature_dimension(), 2);
  EXPECT_THAT(params.dimension_numbers().kernel_spatial_dimensions(),
              ElementsAreArray({0, 1}));
  EXPECT_EQ(params.dimension_numbers().output_batch_dimension(), 0);
  EXPECT_EQ(params.dimension_numbers().output_feature_dimension(), 3);
  EXPECT_THAT(params.dimension_numbers().output_spatial_dimensions(),
              ElementsAreArray({1, 2}));
}

TEST(UniformQuantizedConvolutionParamsTest, CalculateOutputShapeDefaultAttr) {
  UniformQuantizedConvolutionDimensionNumbersAttr dimension_numbers;
  UniformQuantizedConvolutionParams params(/*window_strides=*/{},
                                           /*lhs_dilation=*/{},
                                           /*rhs_dilation=*/{},
                                           dimension_numbers,
                                           /*feature_group_count=*/1,
                                           /*batch_group_count=*/1,
                                           /*padding=*/"VALID");

  const TensorShape lhs_shape({2, 2, 3, 4});
  const TensorShape rhs_shape({3, 2, 2, 3});
  TF_ASSERT_OK(
      params.ValidateOrFillParamsAndValidateShape(lhs_shape, rhs_shape));

  auto shape_or = params.CalculateOutputShape(lhs_shape, rhs_shape);
  TF_ASSERT_OK(shape_or.status());
  EXPECT_TRUE(shape_or.value().IsSameSize({2, 3, 2, 2}));
}

TEST(UniformQuantizedConvolutionParamsTest, CalculateOutputShapeSetAttr) {
  UniformQuantizedConvolutionDimensionNumbersAttr dimension_numbers;
  ASSERT_TRUE(TextFormat::ParseFromString(R"pb(
                                            input_batch_dimension: 0
                                            input_feature_dimension: 3
                                            input_spatial_dimensions: 1
                                            input_spatial_dimensions: 2
                                            kernel_output_feature_dimension: 3
                                            kernel_input_feature_dimension: 2
                                            kernel_spatial_dimensions: 0
                                            kernel_spatial_dimensions: 1
                                            output_batch_dimension: 0
                                            output_feature_dimension: 3
                                            output_spatial_dimensions: 1
                                            output_spatial_dimensions: 2
                                          )pb",
                                          &dimension_numbers));
  UniformQuantizedConvolutionParams params(/*window_strides=*/{2, 2},
                                           /*lhs_dilation=*/{3, 3},
                                           /*rhs_dilation=*/{4, 4},
                                           dimension_numbers,
                                           /*feature_group_count=*/2,
                                           /*batch_group_count=*/1,
                                           /*padding=*/"EXPLICIT",
                                           /*padding_list=*/{1, 1, 2, 2});
  const TensorShape lhs_shape({2, 3, 4, 2});
  const TensorShape rhs_shape({2, 3, 1, 2});
  TF_ASSERT_OK(
      params.ValidateOrFillParamsAndValidateShape(lhs_shape, rhs_shape));

  auto shape_or = params.CalculateOutputShape(lhs_shape, rhs_shape);
  TF_ASSERT_OK(shape_or.status());
  EXPECT_TRUE(shape_or.value().IsSameSize({2, 3, 3, 2}));
}

TEST(UniformQuantizedConvolutionParamsTest, CalculateSameOptionPadding) {
  UniformQuantizedConvolutionDimensionNumbersAttr dimension_numbers;
  UniformQuantizedConvolutionParams params(/*window_strides=*/{},
                                           /*lhs_dilation=*/{},
                                           /*rhs_dilation=*/{},
                                           dimension_numbers,
                                           /*feature_group_count=*/1,
                                           /*batch_group_count=*/1,
                                           /*padding=*/"SAME");

  const TensorShape lhs_shape({2, 2, 3, 4});
  const TensorShape rhs_shape({3, 2, 4, 3});
  TF_ASSERT_OK(
      params.ValidateOrFillParamsAndValidateShape(lhs_shape, rhs_shape));

  EXPECT_THAT(params.padding_list(), ElementsAreArray({1, 2, 1, 1}));
}

}  // namespace
}  // namespace machina
