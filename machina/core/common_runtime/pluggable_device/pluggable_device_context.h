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

#ifndef MACHINA_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_CONTEXT_H_
#define MACHINA_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_CONTEXT_H_

#include <functional>

#include "absl/status/status.h"
#include "machina/core/common_runtime/device.h"
#include "machina/core/framework/device_base.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/lib/gtl/inlined_vector.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/stringpiece.h"

namespace stream_executor {
class Stream;
}  // namespace stream_executor

namespace machina {

class PluggableDeviceContext : public DeviceContext {
 public:
  // Does not take ownership of streams.
  PluggableDeviceContext(
      int stream_id, se::Stream* stream, se::Stream* host_to_device_stream,
      se::Stream* device_to_host_stream,
      absl::InlinedVector<se::Stream*, 4UL> device_to_device_stream)
      : stream_id_(stream_id),
        stream_(stream),
        host_to_device_stream_(host_to_device_stream),
        device_to_host_stream_(device_to_host_stream),
        device_to_device_stream_(device_to_device_stream) {}

  ~PluggableDeviceContext() override = default;

  se::Stream* stream() const override { return stream_; }
  se::Stream* host_to_device_stream() const { return host_to_device_stream_; }
  se::Stream* device_to_host_stream() const { return device_to_host_stream_; }
  se::Stream* device_to_device_stream(int index) const {
    return device_to_device_stream_[index % device_to_device_stream_.size()];
  }
  int stream_id() const { return stream_id_; }

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor, StatusCallback done,
                             bool sync_dst_compute) const override;

  void CopyDeviceTensorToCPU(const Tensor* device_tensor,
                             absl::string_view tensor_name, Device* device,
                             Tensor* cpu_tensor, StatusCallback done) override;

  void CopyTensorInSameDevice(const Tensor* input_tensor, Device* device,
                              Tensor* output_tensor,
                              StatusCallback done) const override;

  void MaintainLifetimeOnStream(const Tensor* t,
                                se::Stream* stream) const override {}

  absl::Status ThenExecute(Device* device, se::Stream* stream,
                           std::function<void()> func) override;

  bool IsPluggableDevice() override;

 private:
  int stream_id_;
  // The default primary stream to use for this context.
  // All the memory belongs to this stream.
  se::Stream* stream_;
  // The stream to use for copying data from host into PluggableDevice.
  se::Stream* host_to_device_stream_;
  // The stream to use for copying data from PluggableDevice to host.
  se::Stream* device_to_host_stream_;
  // Streams to use for copying data between PluggableDevices.
  absl::InlinedVector<se::Stream*, 4UL> device_to_device_stream_;
};

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_CONTEXT_H_
