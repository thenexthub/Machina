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
#include "machina/lite/experimental/shlo/ops/abs.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/lite/experimental/shlo/ops/test_util.h"
#include "machina/lite/experimental/shlo/ops/unary_elementwise_test_util.h"
#include "machina/lite/experimental/shlo/quantize.h"
#include "machina/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "machina/lite/experimental/shlo/shape.h"
#include "machina/lite/experimental/shlo/status_matcher.h"
#include "machina/lite/experimental/shlo/tensor.h"

using testing::ElementsAreArray;

namespace shlo_ref {

template <>
struct ParamName<AbsOp> {
  static std::string Get() { return "Abs"; }
};

namespace {

constexpr struct AbsRef {
  template <class T>
  T operator()(T v) const {
    return v < static_cast<T>(0) ? static_cast<T>(-v) : v;
  }
} abs_ref;

INSTANTIATE_TYPED_TEST_SUITE_P(Abs, UnaryElementwiseOpShapePropagationTest,
                               AbsOp, TestParamNames);

INSTANTIATE_TYPED_TEST_SUITE_P(
    Abs, UnaryElementwiseSameBaselineElementTypeConstraintTest,
    UnaryElementwiseConstraint1Types<AbsOp>, TestParamNames);

using UnsupportedTypes =
    WithOpTypes<AbsOp, ConcatTypes<BoolTestType, PerAxisQuantizedTestTypes>>;

INSTANTIATE_TYPED_TEST_SUITE_P(Abs, UnaryElementwiseUnsupportedTypeTest,
                               UnsupportedTypes, TestParamNames);

template <class T>
struct AbsTest : ::testing::Test {};

TYPED_TEST_SUITE(AbsTest, ArithmeticTestTypes, TestParamNames);

TYPED_TEST(AbsTest, ArithmeticTensorsWork) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape({2, 3, 4});
  Vector<StorageT> input_data = RandomBuffer<TypeParam::kStorage>(shape);
  Vector<StorageT> output_data(shape.NumElements());

  Tensor input_tensor{
      .type = TensorType{.shape = shape, .element_type = TypeParam::kStorage},
      .data = input_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  Vector<StorageT> expected_data(shape.NumElements());
  absl::c_transform(input_data, expected_data.begin(), abs_ref);

  auto op = Create(AbsOp::Attributes{});
  ASSERT_OK(Prepare(op, input_tensor, output_tensor));
  ASSERT_OK(Evaluate(op, input_tensor, output_tensor));
  EXPECT_THAT(output_data, ElementsAreArray(expected_data));
}

template <class T>
struct QuantizedAbsTest : ::testing::Test {};

TYPED_TEST_SUITE(QuantizedAbsTest, QuantizedTestTypes, TestParamNames);

TYPED_TEST(QuantizedAbsTest, QuantizedPerTensor) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape({2, 3, 4});
  Vector<StorageT> input_data = RandomBuffer<TypeParam::kStorage>(shape);
  Vector<StorageT> output_data(shape.NumElements());
  const ExpressedT scale = static_cast<ExpressedT>(1.5);
  const StorageT zero_point = static_cast<StorageT>(5);
  const QuantizedElementTypePerTensor tensor_type =
      QuantizedElementTypePerTensor(TypeParam::kStorage, zero_point,
                                    TypeParam::kExpressed, scale);
  Tensor input_tensor{
      .type = QuantizedPerTensorTensorType{.shape = shape,
                                           .element_type = tensor_type},
      .data = input_data.data()};
  Tensor output_tensor{
      .type = QuantizedPerTensorTensorType{.shape = shape,
                                           .element_type = tensor_type},
      .data = output_data.data()};

  Vector<StorageT> expected_data(shape.NumElements());
  absl::c_transform(
      input_data, expected_data.begin(), [zero_point, scale](auto v) {
        const ExpressedT dequantized_input = Dequantize(v, zero_point, scale);
        const ExpressedT dequantized_res = abs_ref(dequantized_input);
        return Quantize<TypeParam::kStorage, TypeParam::kExpressed>(
            dequantized_res, zero_point, static_cast<ExpressedT>(1.) / scale);
      });

  auto op = Create(AbsOp::Attributes{});
  ASSERT_OK(Prepare(op, input_tensor, output_tensor));
  ASSERT_OK(Evaluate(op, input_tensor, output_tensor));
  EXPECT_THAT(output_data, ElementsAreArray(expected_data));
}

}  // namespace
}  // namespace shlo_ref
