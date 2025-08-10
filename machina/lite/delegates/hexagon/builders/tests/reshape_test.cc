/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/lite/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"
#include "machina/lite/kernels/reshape_test_common.h"

namespace tflite {
using ::testing::ElementsAreArray;

template <typename T>
class ReshapeOpTest : public ::testing::Test {};

using DataTypes = ::testing::Types<uint8_t, int8_t>;
TYPED_TEST_SUITE(ReshapeOpTest, DataTypes);

TYPED_TEST(ReshapeOpTest, RegularShapes) {
  std::vector<ShapeSpecificationType> shape_types = {
      ShapeSpecificationType::kAsReshapeOption,
      ShapeSpecificationType::kAsConstantTensor};

  for (ShapeSpecificationType shape_type : shape_types) {
    ReshapeOpModel<TypeParam, SingleOpModelWithHexagon> m(
        {1, 2, 4, 1}, {3}, {2, 2, 2}, shape_type);
    m.SetInput({1, 2, 3, 4, 5, 6, 7, 8});
    m.ApplyDelegateAndInvoke();
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8}));
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2}));
  }
}

TYPED_TEST(ReshapeOpTest, WithStretchDimension) {
  std::vector<ShapeSpecificationType> shape_types = {
      ShapeSpecificationType::kAsReshapeOption,
      ShapeSpecificationType::kAsConstantTensor};

  for (ShapeSpecificationType shape_type : shape_types) {
    ReshapeOpModel<TypeParam, SingleOpModelWithHexagon> m(
        {1, 2, 4, 1}, {3}, {2, 1, -1}, shape_type);
    m.SetInput({1, 2, 3, 4, 5, 6, 7, 8});
    m.ApplyDelegateAndInvoke();
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8}));
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 4}));
  }
}

}  // namespace tflite
