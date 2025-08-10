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

#include "machina/core/framework/op_kernel.h"

#if GOOGLE_CUDA || MACHINA_USE_ROCM
#include "machina/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM

namespace machina {
namespace {

class SyncDeviceOp : public OpKernel {
 public:
  explicit SyncDeviceOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {}

 private:
  SyncDeviceOp(const SyncDeviceOp&) = delete;
  void operator=(const SyncDeviceOp&) = delete;
};

REGISTER_KERNEL_BUILDER(Name("SyncDevice").Device(DEVICE_DEFAULT),
                        SyncDeviceOp);

#if GOOGLE_CUDA || MACHINA_USE_ROCM
class SyncDeviceGpuOp : public OpKernel {
 public:
  explicit SyncDeviceGpuOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const DeviceBase::AcceleratorDeviceInfo* info =
        context->device()->machina_accelerator_device_info();
    if (info && info->stream) {
      OP_REQUIRES_OK(context, info->stream->BlockHostUntilDone());
    }
  }

 private:
  SyncDeviceGpuOp(const SyncDeviceGpuOp&) = delete;
  void operator=(const SyncDeviceGpuOp&) = delete;
};

REGISTER_KERNEL_BUILDER(Name("SyncDevice").Device(DEVICE_GPU), SyncDeviceGpuOp);
#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM

}  // namespace
}  // namespace machina
