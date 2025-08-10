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

#include "machina/core/common_runtime/threadpool_device.h"

#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/test.h"
#include "machina/core/public/session_options.h"

namespace machina {
namespace {

const int kDimSize = 2;

void InitTensor(Tensor* tensor, float value) {
  auto eigen_tensor = tensor->tensor<float, kDimSize>();
  for (int i = 0; i < kDimSize; ++i) {
    for (int j = 0; j < kDimSize; ++j) {
      eigen_tensor(i, j) = value;
    }
  }
}

bool Equal(const Tensor& tensor1, const Tensor& tensor2) {
  auto eigen_tensor1 = tensor1.tensor<float, kDimSize>();
  auto eigen_tensor2 = tensor2.tensor<float, kDimSize>();
  for (int i = 0; i < kDimSize; ++i) {
    for (int j = 0; j < kDimSize; ++j) {
      if (eigen_tensor1(i, j) != eigen_tensor2(i, j)) {
        return false;
      }
    }
  }
  return true;
}

TEST(ThreadPoolDeviceTest, CopyTensor) {
  Tensor input(DT_FLOAT, TensorShape({kDimSize, kDimSize}));
  Tensor output(DT_FLOAT, TensorShape({kDimSize, kDimSize}));
  InitTensor(&input, 1);
  InitTensor(&output, 0);
  ASSERT_FALSE(Equal(input, output));

  ThreadPoolDevice device(SessionOptions(), "/device:CPU:0", Bytes(256),
                          DeviceLocality(), cpu_allocator());
  DeviceContext* device_context = new DeviceContext;
  Notification note;
  device.CopyTensorInSameDevice(&input, &output, device_context,
                                [&note](const absl::Status& s) {
                                  TF_ASSERT_OK(s);
                                  note.Notify();
                                });
  note.WaitForNotification();
  ASSERT_TRUE(Equal(input, output));

  device_context->Unref();
}

}  // namespace
}  // namespace machina
