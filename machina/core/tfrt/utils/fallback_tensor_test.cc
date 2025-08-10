/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#include "machina/core/tfrt/utils/fallback_tensor.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/core/common_runtime/dma_helper.h"
#include "machina/core/framework/tensor_shape.h"

namespace machina {
namespace tfrt_stub {
namespace {

TEST(FallbackTensorTest, ImmutableTensor) {
  int32_t scalar = 123;
  machina::Tensor tensor(scalar);

  auto immutable_tensor = ImmutableTensor::Create(tensor);

  ASSERT_EQ(immutable_tensor.tensor().NumElements(), 1);
  ASSERT_EQ(immutable_tensor.tensor().dtype(), machina::DT_INT32);
  auto flat = immutable_tensor.tensor().flat<int32_t>();
  EXPECT_EQ(flat(0), 123);
  EXPECT_FALSE(immutable_tensor.tensor().RefCountIsOne());
  EXPECT_EQ(tensor.TotalBytes(), immutable_tensor.tensor().TotalBytes());
}

TEST(FallbackTensorTest, StringImmutableTensor) {
  machina::tstring scalar = "string";
  machina::Tensor tensor(scalar);

  auto immutable_tensor = ImmutableTensor::Create(tensor);

  ASSERT_EQ(immutable_tensor.tensor().NumElements(), 1);
  ASSERT_EQ(immutable_tensor.tensor().dtype(), machina::DT_STRING);
  auto flat = immutable_tensor.tensor().flat<machina::tstring>();
  EXPECT_EQ(flat(0), "string");
  EXPECT_FALSE(immutable_tensor.tensor().RefCountIsOne());
  EXPECT_EQ(tensor.TotalBytes(), immutable_tensor.tensor().TotalBytes());
}

TEST(FallbackTensorTest, FallbackTensor) {
  int32_t scalar = 123;
  machina::Tensor tensor(scalar);

  {
    FallbackTensor fallback_tensor(tensor);
    EXPECT_FALSE(fallback_tensor.is_immutable());

    ASSERT_EQ(fallback_tensor.tensor().NumElements(), 1);
    ASSERT_EQ(fallback_tensor.tensor().dtype(), machina::DT_INT32);
    auto flat = fallback_tensor.tensor().flat<int32_t>();
    EXPECT_EQ(flat(0), 123);

    FallbackTensor copy(fallback_tensor);
    FallbackTensor assign;
    assign = fallback_tensor;

    ASSERT_EQ(copy.tensor().NumElements(), 1);
    ASSERT_EQ(copy.tensor().dtype(), machina::DT_INT32);
    EXPECT_EQ(copy.tensor().flat<int32_t>()(0), 123);
    ASSERT_EQ(assign.tensor().NumElements(), 1);
    ASSERT_EQ(assign.tensor().dtype(), machina::DT_INT32);
    EXPECT_EQ(assign.tensor().flat<int32_t>()(0), 123);

    fallback_tensor = {};

    ASSERT_EQ(copy.tensor().NumElements(), 1);
    ASSERT_EQ(copy.tensor().dtype(), machina::DT_INT32);
    EXPECT_EQ(copy.tensor().flat<int32_t>()(0), 123);
    ASSERT_EQ(assign.tensor().NumElements(), 1);
    ASSERT_EQ(assign.tensor().dtype(), machina::DT_INT32);
    EXPECT_EQ(assign.tensor().flat<int32_t>()(0), 123);
  }

  auto immutable_tensor = ImmutableTensor::Create(tensor);

  {
    FallbackTensor fallback_tensor(&immutable_tensor);
    EXPECT_TRUE(fallback_tensor.is_immutable());

    ASSERT_EQ(fallback_tensor.tensor().NumElements(), 1);
    ASSERT_EQ(fallback_tensor.tensor().dtype(), machina::DT_INT32);
    auto flat = fallback_tensor.tensor().flat<int32_t>();
    EXPECT_EQ(flat(0), 123);

    FallbackTensor copy(fallback_tensor);
    FallbackTensor assign;
    assign = fallback_tensor;

    ASSERT_EQ(copy.tensor().NumElements(), 1);
    ASSERT_EQ(copy.tensor().dtype(), machina::DT_INT32);
    EXPECT_EQ(copy.tensor().flat<int32_t>()(0), 123);
    ASSERT_EQ(assign.tensor().NumElements(), 1);
    ASSERT_EQ(assign.tensor().dtype(), machina::DT_INT32);
    EXPECT_EQ(assign.tensor().flat<int32_t>()(0), 123);

    fallback_tensor = {};

    ASSERT_EQ(copy.tensor().NumElements(), 1);
    ASSERT_EQ(copy.tensor().dtype(), machina::DT_INT32);
    EXPECT_EQ(copy.tensor().flat<int32_t>()(0), 123);
    ASSERT_EQ(assign.tensor().NumElements(), 1);
    ASSERT_EQ(assign.tensor().dtype(), machina::DT_INT32);
    EXPECT_EQ(assign.tensor().flat<int32_t>()(0), 123);
  }
}

TEST(FallbackTensorTest, FallbackTensorCopy) {
  int32_t scalar = 123;
  machina::Tensor tensor(scalar);

  {
    FallbackTensor fallback_tensor(tensor);
    EXPECT_FALSE(fallback_tensor.is_immutable());

    auto copy = fallback_tensor;
    EXPECT_TRUE(copy.is_immutable());
  }

  auto immutable_tensor = ImmutableTensor::Create(tensor);

  {
    FallbackTensor fallback_tensor(&immutable_tensor);
    EXPECT_TRUE(fallback_tensor.is_immutable());

    auto copy = fallback_tensor;
    EXPECT_TRUE(copy.is_immutable());
  }
}

TEST(FallbackTensorTest, FallbackTensorCopyRootBuffer) {
  int32_t scalar = 123;
  machina::Tensor tensor(scalar);
  auto immutable_tensor = ImmutableTensor::Create(tensor);

  FallbackTensor fallback_tensor(&immutable_tensor);
  EXPECT_TRUE(fallback_tensor.is_immutable());

  EXPECT_EQ(fallback_tensor.buffer()->root_buffer(),
            machina::DMAHelper::buffer(&tensor));

  FallbackTensor copy = fallback_tensor;
  EXPECT_TRUE(copy.is_immutable());

  EXPECT_EQ(copy.buffer()->root_buffer(),
            machina::DMAHelper::buffer(&tensor));
}

TEST(FallbackTensorTest, EmptyTensor) {
  machina::Tensor tensor(machina::DT_FLOAT,
                            machina::TensorShape({1, 0}));

  FallbackTensor fallback_tensor(tensor);
  auto copy = fallback_tensor;
  ASSERT_FALSE(copy.buffer());
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace machina
