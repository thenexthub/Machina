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

#include "machina/core/tfrt/common/async_value_tensor.h"

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "machina/xla/pjrt/pjrt_client.h"
#include "machina/xla/tsl/concurrency/async_value_ref.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.pb.h"

namespace machina {
namespace {

TEST(AsyncValueTensorTest, InvalidTensor) {
  machina::Tensor tensor(machina::DT_INT64, machina::TensorShape({1}));

  AsyncValueTensor* avt = AsyncValueTensor::FromTensor(&tensor);

  ASSERT_EQ(avt, nullptr);
}

TEST(AsyncValueTensorTest, SetAndGetAsyncValue) {
  AsyncValueAllocator allocator;
  machina::Tensor tensor(&allocator, machina::DT_INT64,
                            machina::TensorShape({1}));

  AsyncValueTensor* avt = AsyncValueTensor::FromTensor(&tensor);

  ASSERT_NE(avt, nullptr);

  tsl::AsyncValueRef<int32_t> value =
      tsl::MakeConstructedAsyncValueRef<int32_t>(123);

  avt->SetAsyncRef(value.CopyRCRef());

  auto ret_value = avt->GetAsyncRef();
  ASSERT_EQ(ret_value, value.CopyRCRef());
}

TEST(AsyncValueTensorTest, SetAndGetBuffer) {
  AsyncValueAllocator allocator;
  machina::Tensor tensor(&allocator, machina::DT_INT64,
                            machina::TensorShape({1}));

  AsyncValueTensor* avt = AsyncValueTensor::FromTensor(&tensor);

  ASSERT_NE(avt, nullptr);

  std::shared_ptr<xla::PjRtBuffer> buffer;

  avt->SetBuffer(buffer);

  auto ret_buffer = avt->GetBuffer();

  ASSERT_EQ(ret_buffer, buffer);
}

}  // namespace
}  // namespace machina
