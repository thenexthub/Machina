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
#include "machina/lite/kernels/unsorted_segment_test.h"

#include <limits.h>
#include <stdint.h>

#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>
#include "machina/lite/kernels/test_util.h"
#include "machina/lite/schema/schema_generated.h"

namespace tflite {
namespace {

TEST_P(UnsortedSegmentTest, SegmentIdsSizeNotEqualToDataFirstDimensionFails) {
  UnsortedSegmentModel<int32_t> model =
      getModel({TensorType_INT32, {3, 2}}, {TensorType_INT32, {2}},
               {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1});
  model.PopulateTensor<int32_t>(model.num_segments(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteError);
}
TEST_P(UnsortedSegmentTest,
       LargestSegmentIdPlusOneGreaterThanNumSegmentsFails) {
  UnsortedSegmentModel<int32_t> model =
      getModel({TensorType_INT32, {2, 2}}, {TensorType_INT32, {2}},
               {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1});
  model.PopulateTensor<int32_t>(model.num_segments(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteError);
}
TEST_P(UnsortedSegmentTest, NumSegmentsNotScalarShapeFails) {
  UnsortedSegmentModel<int32_t> model =
      getModel({TensorType_INT32, {3, 2}}, {TensorType_INT32, {3}},
               {TensorType_INT32, {2}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1, 0});
  model.PopulateTensor<int32_t>(model.num_segments(), {2, 1});
  ASSERT_EQ(model.Invoke(), kTfLiteError);
}
TEST_P(UnsortedSegmentTest, Rank2SegIdsNotPrefixFails) {
  UnsortedSegmentModel<int32_t> model =
      getModel({TensorType_INT32, {2, 2, 2}}, {TensorType_INT32, {2, 1}},
               {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int32_t>(model.segment_ids(), {1, 1});
  model.PopulateTensor<int32_t>(model.num_segments(), {3});
  ASSERT_EQ(model.Invoke(), kTfLiteError);
}
TEST_P(UnsortedSegmentTest, Rank2SegIdsHasShapeNumSegDataShapeSuffix) {
  UnsortedSegmentModel<int32_t> model =
      getModel({TensorType_INT32, {2, 2, 2}}, {TensorType_INT32, {2, 2}},
               {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int32_t>(model.segment_ids(), {1, 2, 0, 8});
  model.PopulateTensor<int32_t>(model.num_segments(), {10});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), testing::ElementsAreArray({10, 2}));
}
TEST_P(UnsortedSegmentTest, Rank2SegIdsHasShapeNumSegDataShapeSuffixConst) {
  UnsortedSegmentModel<int32_t> model = getConstModel(
      {TensorType_INT32, {2, 2, 2}}, {1, 2, -1, -1}, {2, 2}, {3}, {1});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4, 5, 6});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), testing::ElementsAreArray({3, 2}));
}
TEST_P(UnsortedSegmentTest, SegIdsHasSameShapeAsData2d) {
  UnsortedSegmentModel<int32_t> model =
      getModel({TensorType_INT32, {2, 2}}, {TensorType_INT32, {2, 2}},
               {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1, 5, 2, 4});
  model.PopulateTensor<int32_t>(model.num_segments(), {10});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), testing::ElementsAreArray({10}));
}
TEST_P(UnsortedSegmentTest, SegIdsHasSameShapeAsData2dConst) {
  UnsortedSegmentModel<int32_t> model =
      getConstModel({TensorType_INT32, {2, 2}}, {1, 1, 1, 1}, {2, 2}, {3}, {1});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), testing::ElementsAreArray({3}));
}
TEST_P(UnsortedSegmentTest, SegIdsHasSameShapeAsData3d) {
  UnsortedSegmentModel<int32_t> model =
      getModel({TensorType_INT32, {2, 2, 2}}, {TensorType_INT32, {2, 2, 2}},
               {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4, 5, 6, 7, 8});
  model.PopulateTensor<int32_t>(model.segment_ids(), {1, 2, 3, 4, 5, 6, 7, 8});
  model.PopulateTensor<int32_t>(model.num_segments(), {10});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), testing::ElementsAreArray({10}));
}
TEST_P(UnsortedSegmentTest, SegIdsHasSameShapeAsData3dConst) {
  UnsortedSegmentModel<int32_t> model =
      getConstModel({TensorType_INT32, {2, 2, 2}}, {0, 1, 2, -1, 3, -1, 4, -1},
                    {2, 2, 2}, {8}, {1});
  model.PopulateTensor<int32_t>(model.data(), {1, 1, 1, 1, 1, 1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), testing::ElementsAreArray({8}));
}
TEST_P(UnsortedSegmentTest, Data5dHasShapeNumSegDataShapeSuffix) {
  UnsortedSegmentModel<int32_t> model =
      getModel({TensorType_INT32, {2, 1, 2, 1, 2}}, {TensorType_INT32, {2, 1}},
               {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4, 5, 6, 7, 8});
  model.PopulateTensor(model.segment_ids(), {0, 1});
  model.PopulateTensor(model.num_segments(), {10});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), testing::ElementsAreArray({10, 2, 1, 2}));
}
}  // namespace
}  // namespace tflite
