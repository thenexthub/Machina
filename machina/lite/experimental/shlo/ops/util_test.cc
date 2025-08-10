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
#include "machina/lite/experimental/shlo/ops/util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "machina/lite/experimental/shlo/shape.h"
#include "machina/lite/experimental/shlo/status_matcher.h"

using testing::ElementsAreArray;

namespace shlo_ref {
namespace {

TEST(UtilTest, PropagateToEmptyShape) {
  const Shape input_shape({2, 3, 4, 5});
  Shape output_shape;

  EXPECT_OK(Propagate(input_shape, output_shape));
  EXPECT_THAT(output_shape.Dimensions(),
              ElementsAreArray(input_shape.Dimensions()));
}

TEST(UtilTest, PropagateToIncompatibleShapeFails) {
  const Shape input_shape({2, 3, 4, 5});
  Shape output_shape({2, 3, 4, 6});

  EXPECT_THAT(
      Propagate(input_shape, output_shape),
      absl::FailedPreconditionError(
          "The specified output tensor shape is not compatible with the input "
          "shape."));
  EXPECT_THAT(output_shape.Dimensions(), ElementsAreArray({2, 3, 4, 6}));
}

}  // namespace
}  // namespace shlo_ref
