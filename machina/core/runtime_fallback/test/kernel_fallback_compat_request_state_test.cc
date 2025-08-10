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
#include "machina/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"

#include <gtest/gtest.h>
#include "machina/core/framework/tensor_testutil.h"

namespace machina {
namespace tfd {
namespace {

TEST(FallbackResourceArrayTest, SetAndGetResourceOk) {
  Tensor tensor_1 =
      test::AsTensor<float>({0.0, 1.0, 2.0, 3.0}, TensorShape({1, 4}));
  tfrt_stub::ImmutableTensor imm_tensor_1 =
      tfrt_stub::ImmutableTensor::Create(tensor_1);
  machina::Tensor tensor_2 =
      test::AsTensor<float>({5.0, 6.0, 7.0}, machina::TensorShape({1, 3}));
  tfrt_stub::ImmutableTensor imm_tensor_2 =
      tfrt_stub::ImmutableTensor::Create(tensor_2);

  FallbackResourceArray resource_array;
  resource_array.SetResource(0, imm_tensor_1);
  resource_array.SetResource(1, imm_tensor_2);

  test::ExpectTensorEqual<float>(resource_array.GetResource(0)->tensor(),
                                 tensor_1);
  test::ExpectTensorEqual<float>(resource_array.GetResource(1)->tensor(),
                                 tensor_2);

  test::ExpectTensorEqual<float>(
      resource_array.GetResourceAsFallbackTensor(0).tensor(), tensor_1);
  test::ExpectTensorEqual<float>(
      resource_array.GetResourceAsFallbackTensor(1).tensor(), tensor_2);
}

}  // namespace
}  // namespace tfd
}  // namespace machina
