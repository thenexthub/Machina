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

#ifndef MACHINA_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_H_
#define MACHINA_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "machina/xla/stream_executor/stream_executor.h"
#include "machina/core/common_runtime/device/device_event_mgr.h"
#include "machina/core/common_runtime/device/device_id.h"
#include "machina/core/common_runtime/device/device_id_manager.h"
#include "machina/core/common_runtime/local_device.h"
#include "machina/core/common_runtime/pluggable_device/pluggable_device_context.h"
#include "machina/core/common_runtime/shared_counter.h"
#include "machina/core/framework/allocator.h"
#include "machina/core/framework/device_attributes.pb.h"
#include "machina/core/framework/device_base.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/graph/types.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/lib/gtl/inlined_vector.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/platform/stream_executor.h"
#include "machina/core/platform/threadpool.h"
#include "machina/core/platform/types.h"
#include "machina/core/public/session_options.h"

namespace machina {

class PluggableDevice : public LocalDevice {
 public:
  PluggableDevice(const SessionOptions& options, const std::string& name,
                  const string& device_type, const string& platform_name,
                  Bytes memory_limit, const DeviceLocality& locality,
                  TfDeviceId tf_device_id,
                  const std::string& physical_device_desc,
                  Allocator* device_allocator, Allocator* cpu_allocator,
                  bool sync_every_op);

  ~PluggableDevice() override;

  // Initialize the device and return the status of initialization.
  absl::Status Init(const SessionOptions& options);

  void ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                    AsyncOpKernel::DoneCallback done) override;

  void Compute(OpKernel* op_kernel, OpKernelContext* context) override;

  absl::Status Sync() override;

  Allocator* GetAllocator(AllocatorAttributes attr) override;

  absl::Status MakeTensorFromProto(const TensorProto& tensor_proto,
                                   AllocatorAttributes alloc_attrs,
                                   Tensor* tensor) override;

  void CopyTensorInSameDevice(const Tensor* input_tensor, Tensor* output_tensor,
                              const DeviceContext* device_context,
                              StatusCallback done) override;

  // The executor that provides control for the pluggable device;
  se::StreamExecutor* executor() const { return executor_; }

 private:
  Allocator* device_allocator_;
  Allocator* cpu_allocator_;

  se::StreamExecutor* executor_ = nullptr;
  struct StreamGroup {
    se::Stream* compute = nullptr;
    se::Stream* host_to_device = nullptr;
    se::Stream* device_to_host = nullptr;
    absl::InlinedVector<se::Stream*, 4UL> device_to_device;
  };

  class StreamGroupFactory;

  StreamGroup* stream_;
  PluggableDeviceContext* device_context_;
  // TODO(penpornk): Investigate renaming `GpuDeviceInfo` to `DeviceInfo`.
  DeviceBase::AcceleratorDeviceInfo* pluggable_device_info_ = nullptr;
  TfDeviceId tf_device_id_;
  const string platform_name_;
  const bool sync_every_op_ = false;
  EventMgr* em_ = nullptr;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  bool force_gpu_compatible_ = false;
  std::string ComputeOpKernelDebugString(const OpKernel& op_kernel,
                                         int stream_id);

  // This method returns an initialization status, in addition to
  // calling the "done" StatusCallback, if there is a failure to
  // allocate memory or if the tensor "from" is not DMA-copyable.
  // If there is no error prior to enqueueing the copy, an OK status
  // is returned.
  absl::Status MaybeCopyTensorToPluggableDevice(
      const AllocatorAttributes& alloc_attrs, const Tensor& from, Tensor* to,
      StatusCallback done);
};

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_H_
