/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#ifndef MACHINA_CORE_COMMON_RUNTIME_RENAMED_DEVICE_H_
#define MACHINA_CORE_COMMON_RUNTIME_RENAMED_DEVICE_H_

#include "machina/core/common_runtime/device.h"
#include "machina/core/lib/core/threadpool_interface.h"
#include "machina/core/util/device_name_utils.h"

namespace machina {

// Wraps a device with a new name, delegating work to the wrapped device.
//
// This class is used to wrap local devices when using clusterspec propagation
// where the name of a particular device may change in the context of a given
// session.
class RenamedDevice : public Device {
 public:
  static std::unique_ptr<Device> NewRenamedDevice(
      const string& new_base, Device* underlying, bool owns_underlying,
      bool isolate_session_state,
      thread::ThreadPoolInterface* underlying_threadpool = nullptr);

  ~RenamedDevice() override;

  const DeviceBase* UnderlyingDevice() const override {
    return underlying_device_->UnderlyingDevice();
  }
  DeviceBase* UnderlyingDevice() override {
    return underlying_device_->UnderlyingDevice();
  }

  const CpuWorkerThreads* machina_cpu_worker_threads() const override {
    if (underlying_threadpool_) {
      return Device::machina_cpu_worker_threads();
    }
    return underlying_device_->machina_cpu_worker_threads();
  }

  const DeviceBase::AcceleratorDeviceInfo* machina_accelerator_device_info()
      const override {
    return underlying_device_->machina_accelerator_device_info();
  }

  Allocator* GetAllocator(AllocatorAttributes attr) override {
    return underlying_device_->GetAllocator(attr);
  }

  Allocator* GetScopedAllocator(AllocatorAttributes attr,
                                int64_t step_id) override {
    return underlying_device_->GetScopedAllocator(attr, step_id);
  }

  ScopedAllocatorMgr* GetScopedAllocatorMgr() const override {
    return underlying_device_->GetScopedAllocatorMgr();
  }

  const Eigen::ThreadPoolDevice* eigen_cpu_device() override {
    // Use the underlying threadpool only if the underlying device supports
    // eigen_cpu_device.
    if (underlying_threadpool_ && underlying_device_->has_eigen_cpu_device()) {
      return Device::eigen_cpu_device();
    }
    return underlying_device_->eigen_cpu_device();
  }

  thread::ThreadPool* machina_device_thread_pool() override {
    // Use the underlying threadpool instead of machina_device_thread_pool
    // of the underlying device only if machina_device_thread_pool is defined
    // for the underlying device.
    if (underlying_threadpool_ &&
        underlying_device_->machina_device_thread_pool() != nullptr) {
      return Device::machina_device_thread_pool();
    }
    return underlying_device_->machina_device_thread_pool();
  }

  bool has_eigen_cpu_device() const override {
    return underlying_device_->has_eigen_cpu_device();
  }


  PerOpGpuDevice* MakeGpuDevice() override {
    return underlying_device_->MakeGpuDevice();
  }

  absl::Status ReinitializeGpuDevice(OpKernelContext* context,
                                     PerOpGpuDevice* device, DeviceContext* dc,
                                     Allocator* allocator) override {
    return underlying_device_->ReinitializeGpuDevice(context, device, dc,
                                                     allocator);
  }

  absl::Status MakeTensorFromProto(const TensorProto& tensor_proto,
                                   const AllocatorAttributes alloc_attrs,
                                   Tensor* tensor) override {
    return underlying_device_->MakeTensorFromProto(tensor_proto, alloc_attrs,
                                                   tensor);
  }

  void CopyTensorInSameDevice(const Tensor* input_tensor, Tensor* output_tensor,
                              const DeviceContext* device_context,
                              StatusCallback done) override {
    underlying_device_->CopyTensorInSameDevice(input_tensor, output_tensor,
                                               device_context, std::move(done));
  }

  // Below are virtual methods defined on Device

  void Compute(OpKernel* op_kernel, OpKernelContext* context) override {
    underlying_device_->Compute(op_kernel, context);
  }

  void ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                    AsyncOpKernel::DoneCallback done) override {
    underlying_device_->ComputeAsync(op_kernel, context, std::move(done));
  }

  absl::Status Sync() override { return underlying_device_->Sync(); }

  absl::Status MaybeRewriteGraph(std::unique_ptr<Graph>* graph) override {
    return underlying_device_->MaybeRewriteGraph(graph);
  }

  absl::Status TryGetDeviceContext(DeviceContext** out_context) override {
    return underlying_device_->TryGetDeviceContext(out_context);
  }

  // Returns the resource manager associated w/ this device.
  ResourceMgr* resource_manager() override {
    if (isolate_session_state_) {
      return Device::resource_manager();
    } else {
      return underlying_device_->resource_manager();
    }
  }

  bool IsLocal() const override { return underlying_device_->IsLocal(); }

  bool IsRemoteCallAllowed() const override {
    return underlying_device_->IsRemoteCallAllowed();
  }

 private:
  RenamedDevice(Device* underlying, const DeviceAttributes& attributes,
                bool owns_underlying, bool isolate_session_state,
                thread::ThreadPoolInterface* underlying_threadpool);
  Device* const underlying_device_;
  const bool owns_underlying_device_;
  const bool isolate_session_state_;

  std::unique_ptr<thread::ThreadPool> underlying_threadpool_;
  // eigen_worker_threads_ is stored here so that we can pass the pointer
  // of eigen_worker_threads_.workers to the parent class.
  DeviceBase::CpuWorkerThreads eigen_worker_threads_;
};

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_RENAMED_DEVICE_H_
