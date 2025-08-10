/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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
#ifndef MACHINA_CORE_TFRT_FALLBACK_DEVICE_WITH_CUSTOM_ALLOCATOR_H_
#define MACHINA_CORE_TFRT_FALLBACK_DEVICE_WITH_CUSTOM_ALLOCATOR_H_

#include <utility>

#include "machina/xla/tsl/framework/allocator.h"
#include "machina/core/framework/device.h"

namespace machina {
namespace tfrt_stub {

class DeviceWithCustomAllocator : public machina::Device {
 public:
  DeviceWithCustomAllocator(machina::Device* device,
                            machina::Allocator* allocator)
      : Device(device->env(), device->attributes()),
        device_(device),
        allocator_(allocator) {
    DCHECK(device_);
    DCHECK(allocator_);
  }

  Allocator* GetAllocator(AllocatorAttributes attr) override {
    return allocator_;
  }

  const DeviceBase* UnderlyingDevice() const override {
    return device_->UnderlyingDevice();
  }
  DeviceBase* UnderlyingDevice() override {
    return device_->UnderlyingDevice();
  }

  const CpuWorkerThreads* machina_cpu_worker_threads() const override {
    return device_->machina_cpu_worker_threads();
  }

  Allocator* GetScopedAllocator(AllocatorAttributes attr,
                                int64_t step_id) override {
    return device_->GetScopedAllocator(attr, step_id);
  }

  ScopedAllocatorMgr* GetScopedAllocatorMgr() const override {
    return device_->GetScopedAllocatorMgr();
  }

  const Eigen::ThreadPoolDevice* eigen_cpu_device() override {
    return device_->eigen_cpu_device();
  }

  thread::ThreadPool* machina_device_thread_pool() override {
    return device_->machina_device_thread_pool();
  }

  bool has_eigen_cpu_device() const override {
    return device_->has_eigen_cpu_device();
  }

  absl::Status MakeTensorFromProto(const TensorProto& tensor_proto,
                                   const AllocatorAttributes alloc_attrs,
                                   Tensor* tensor) override {
    return device_->MakeTensorFromProto(tensor_proto, alloc_attrs, tensor);
  }

  void CopyTensorInSameDevice(const Tensor* input_tensor, Tensor* output_tensor,
                              const DeviceContext* device_context,
                              StatusCallback done) override {
    device_->CopyTensorInSameDevice(input_tensor, output_tensor, device_context,
                                    std::move(done));
  }

  absl::Status Sync() override { return device_->Sync(); }

  // Returns the resource manager associated w/ this device.
  ResourceMgr* resource_manager() override {
    return device_->resource_manager();
  }

 private:
  machina::Device* device_ = nullptr;
  machina::Allocator* allocator_ = nullptr;
};

}  // namespace tfrt_stub
}  // namespace machina

#endif  // MACHINA_CORE_TFRT_FALLBACK_DEVICE_WITH_CUSTOM_ALLOCATOR_H_
