/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

// Op kernels used to swap data in and out of GPU memory.

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "machina/core/common_runtime/device.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/lib/core/status.h"

namespace machina {
namespace {

class CopyFromGpuToHostKernel : public AsyncOpKernel {
 public:
  explicit CopyFromGpuToHostKernel(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    const Tensor& input = ctx->input(0);
    OP_REQUIRES_ASYNC(
        ctx, !ctx->input_alloc_attr(0).on_host(),
        absl::InternalError("The input tensor to the _CopyFromGpuToHost kernel "
                            "must reside on the device."),
        done);

    AllocatorAttributes alloc_attrs;
    alloc_attrs.set_gpu_compatible(true);
    alloc_attrs.set_on_host(true);
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->allocate_output(0, input.shape(), &output, alloc_attrs),
        done);

    ctx->op_device_context()->CopyDeviceTensorToCPU(
        &input, "CopyFromGpuToHost", static_cast<Device*>(ctx->device()),
        output, [ctx, done](const absl::Status& s) {
          ctx->SetStatus(s);
          done();
        });
  }
};

REGISTER_KERNEL_BUILDER(
    Name("_CopyFromGpuToHost").Device(DEVICE_GPU).HostMemory("output"),
    CopyFromGpuToHostKernel);

class CopyFromHostToGpuKernel : public AsyncOpKernel {
 public:
  explicit CopyFromHostToGpuKernel(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    const Tensor& input = ctx->input(0);
    OP_REQUIRES_ASYNC(
        ctx, ctx->input_alloc_attr(0).on_host(),
        absl::InternalError("The input tensor to the _CopyFromHostToGpu kernel "
                            "must reside on the host."),
        done);

    Tensor* output;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, input.shape(), &output),
                         done);

    ctx->op_device_context()->CopyCPUTensorToDevice(
        &input, static_cast<Device*>(ctx->device()), output,
        [ctx, done](const absl::Status& s) {
          ctx->SetStatus(s);
          done();
        });
  }
};

REGISTER_KERNEL_BUILDER(
    Name("_CopyFromHostToGpu").Device(DEVICE_GPU).HostMemory("input"),
    CopyFromHostToGpuKernel);

}  // namespace
}  // namespace machina
