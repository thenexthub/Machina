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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "machina/lite/c/c_api_types.h"
#include "machina/lite/delegates/xnnpack/transpose_tester.h"
#include "machina/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "machina/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

TEST(UnsignedQuantizedTranspose, 1D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::vector<int32_t> perm{0};
  // clang-format off
  TransposeTester()
      .num_dims(1)
      .input_shape({37})
      .perm(perm)
      .Test(TensorType_INT8, xnnpack_delegate.get());
  // clang-format on
}

TEST(UnsignedQuantizedTranspose, 2D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::vector<int32_t> perm{0, 1};
  do {
    // clang-format off
    TransposeTester()
        .num_dims(2)
        .input_shape({37, 113})
        .perm(perm)
        .Test(TensorType_INT8, xnnpack_delegate.get());
    // clang-format on
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(UnsignedQuantizedTranspose, 3D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::vector<int32_t> perm{0, 1, 2};
  do {
    TransposeTester()
        .num_dims(3)
        .input_shape({5, 7, 11})
        .perm(perm)
        .Test(TensorType_UINT8, xnnpack_delegate.get());
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(UnsignedQuantizedTranspose, 4D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::vector<int32_t> perm{0, 1, 2, 3};
  do {
    TransposeTester()
        .num_dims(4)
        .input_shape({5, 7, 11, 13})
        .perm(perm)
        .Test(TensorType_UINT8, xnnpack_delegate.get());
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(UnsignedQuantizedTranspose, 5D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::vector<int32_t> perm{0, 1, 2, 3, 4};
  do {
    TransposeTester()
        .num_dims(5)
        .input_shape({3, 5, 7, 11, 13})
        .perm(perm)
        .Test(TensorType_UINT8, xnnpack_delegate.get());
  } while (std::next_permutation(perm.begin(), perm.end()));
}

}  // namespace xnnpack
}  // namespace tflite
