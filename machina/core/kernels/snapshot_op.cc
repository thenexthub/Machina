/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

// See docs in ../ops/array_ops.cc.
#include "machina/core/kernels/snapshot_op.h"

#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/register_types.h"
#include "machina/core/framework/types.h"

namespace machina {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename Scalar>
class SnapshotOp : public OpKernel {
 public:
  explicit SnapshotOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    Tensor* output = nullptr;
    // Try to use buffer forwarding to avoid an explicit copy.
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input.shape(), &output));
    if (!output->SharesBufferWith(input)) {
      functor::Snapshot<Device, Scalar> functor;
      functor(context->eigen_device<Device>(), input.flat<Scalar>(),
              output->flat<Scalar>());
    }
  }
};

#define REGISTER_KERNEL(TYPE)                                        \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Snapshot").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      SnapshotOp<CPUDevice, TYPE>);

TF_CALL_POD_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA || MACHINA_USE_ROCM
#define REGISTER_KERNEL(TYPE)                                        \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Snapshot").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      SnapshotOp<GPUDevice, TYPE>);

TF_CALL_POD_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif


}  // namespace machina
