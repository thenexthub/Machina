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
#include "machina/compiler/jit/pjrt_tensor_buffer_util.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "machina/compiler/jit/test_util.h"
#include "machina/xla/pjrt/pjrt_client.h"
#include "machina/xla/shape.h"
#include "machina/xla/shape_util.h"
#include "machina/core/framework/allocator.h"
#include "machina/core/framework/device.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.h"
#include "machina/core/tfrt/common/pjrt_util.h"
#include "tsl/platform/statusor.h"

namespace machina {
namespace {

TEST(PjRtTensorBufferUtilTest, MakeTensorFromPjRtBuffer) {
  DeviceSetup device_setup;
  device_setup.AddDevicesAndSetUp({DEVICE_GPU});
  Device* device = device_setup.GetDevice(DEVICE_GPU);
  std::vector<int64_t> dimensions = {2, 3};
  Tensor dest_cpu_tensor(cpu_allocator(), machina::DT_INT32,
                         machina::TensorShape(dimensions));
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client, GetPjRtClient(DEVICE_GPU));
  std::vector<int32_t> data{1, 2, 3, 4, 5, 6};
  xla::Shape xla_shape = xla::ShapeUtil::MakeShape(xla::S32, dimensions);
  xla::PjRtDevice* pjrt_device = pjrt_client->addressable_devices()[0];
  TF_ASSERT_OK_AND_ASSIGN(xla::PjRtMemorySpace * pjrt_memory,
                          pjrt_device->default_memory_space());
  TF_ASSERT_OK_AND_ASSIGN(
      auto pjrt_buffer,
      pjrt_client->BufferFromHostBuffer(
          data.data(), xla_shape.element_type(), xla_shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          nullptr, pjrt_memory, /*device_layout=*/nullptr));

  TF_ASSERT_OK_AND_ASSIGN(
      Tensor tensor, MakeTensorFromPjRtBuffer(DT_INT32, TensorShape(dimensions),
                                              std::move(pjrt_buffer)));

  auto s = device->machina_accelerator_device_info()
               ->pjrt_context->CopyDeviceTensorToCPUSync(&tensor, "", device,
                                                         &dest_cpu_tensor);
  for (int i = 0; i < tensor.NumElements(); ++i) {
    EXPECT_EQ(dest_cpu_tensor.flat<int32_t>().data()[i], data[i]);
  }
}

}  // namespace
}  // namespace machina
