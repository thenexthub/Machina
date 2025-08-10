/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include "machina/core/kernels/ragged_tensor_to_variant_op_test.h"

#include <vector>

#include <gtest/gtest.h>
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/variant.h"
#include "machina/core/kernels/ops_testutil.h"
#include "machina/core/kernels/ragged_tensor_variant.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace {

// Tests if RaggedTensorToVariant can handle large tensors, whose number of
// elements are more than 2**31 - 1 (the largest positive int32 number)
TEST_F(RaggedTensorToVariantKernelTest, InputSizeOverInt32Range) {
  // ragged_tensor =
  // [[0, 1, ..., num_col - 1],
  //  [0, 1, ..., num_col - 1],
  //  ...
  //  [0, 1, ..., num_col - 1]]
  // shape = {num_row, num_col}
  const int64_t num_row = 1LL << 15;  // 2**15 rows
  const int64_t num_col = 1LL << 16;  // each row has 2**16 values
  const int64_t value_size = num_row * num_col;  // 2**31 values in total

  std::vector<int> batched_values;
  batched_values.reserve(value_size);
  for (int64_t i = 0; i < num_row; ++i) {
    for (int64_t j = 0; j < num_col; ++j) {
      batched_values.emplace_back(j);
    }
  }

  std::vector<int64_t> batched_splits;
  batched_splits.reserve(num_row + 1);
  batched_splits.emplace_back(0);
  int64_t split_value = num_col;
  for (int64_t i = 0; i < num_row; ++i, split_value += num_col) {
    batched_splits.emplace_back(split_value);
  }

  BuildEncodeRaggedTensorGraph<int, int64_t>(
      {batched_splits}, TensorShape({value_size}), batched_values, true);
  TF_ASSERT_OK(RunOpKernel());

  const auto& encoded_list = GetOutput(0)->vec<Variant>();
  EXPECT_EQ(encoded_list.size(), num_row);

  std::vector<int> expectedValues;
  expectedValues.reserve(num_col);
  for (int64_t i = 0; i < num_col; ++i) {
    expectedValues.emplace_back(i);
  }
  RaggedTensorVariant expectedVariant =
      CreateVariantFromRagged<int, int64_t>({}, expectedValues);
  for (int64_t i = 0; i < num_row; ++i) {
    ExpectRaggedTensorVariantEqual<int, int64_t>(
        expectedVariant, *encoded_list(i).get<RaggedTensorVariant>());
  }
}

// Tests if RaggedTensorToVariant can handle large non-ragged tensors, whose
// number of elements are more than 2**31-1 (the largest positive int32 number)
TEST_F(RaggedTensorToVariantKernelTest, NonRaggedInputSizeOverInt32Range) {
  // tensor =
  // [[0, 1, ..., num_col - 1],
  //  [0, 1, ..., num_col - 1],
  //  ...
  //  [0, 1, ..., num_col - 1]]
  // shape = {num_row, num_col}
  const int64_t num_row = 1LL << 15;  // 2**15 rows
  const int64_t num_col = 1LL << 16;  // each row has 2**16 values
  const int64_t value_size = num_row * num_col;  // 2**31 values in total

  std::vector<int> batched_values;
  batched_values.reserve(value_size);
  for (int64_t i = 0; i < num_row; ++i) {
    for (int64_t j = 0; j < num_col; ++j) {
      batched_values.emplace_back(j);
    }
  }
  TensorShape shape({num_row, num_col});
  BuildEncodeRaggedTensorGraph<int, int64_t>({}, shape, batched_values, true);
  TF_ASSERT_OK(RunOpKernel());

  const auto& encoded_list = GetOutput(0)->vec<Variant>();
  EXPECT_EQ(encoded_list.size(), num_row);

  std::vector<int> expectedValues;
  expectedValues.reserve(num_col);
  for (int64_t i = 0; i < num_col; ++i) {
    expectedValues.emplace_back(i);
  }
  RaggedTensorVariant expectedVariant =
      CreateVariantFromRagged<int, int64_t>({}, expectedValues);
  for (int64_t i = 0; i < num_row; ++i) {
    ExpectRaggedTensorVariantEqual<int, int64_t>(
        expectedVariant, *encoded_list(i).get<RaggedTensorVariant>());
  }
}

}  // namespace
}  // namespace machina
