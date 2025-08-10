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

#include "machina/core/kernels/tensor_to_hash_bucket_op.h"

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/register_types.h"

namespace machina {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class TensorToHashBucketOp : public OpKernel {
 public:
  explicit TensorToHashBucketOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_buckets", &num_buckets_));

    DataType dtype;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype));
    OP_REQUIRES(ctx,
                dtype == DT_INT8 || dtype == DT_UINT8 || dtype == DT_INT16 ||
                    dtype == DT_UINT16 || dtype == DT_INT32 ||
                    dtype == DT_UINT32 || dtype == DT_INT64 ||
                    dtype == DT_UINT64,
                errors::InvalidArgument("TensorToHashBucketOp doesn't support "
                                        "datatype ",
                                        DataTypeString(dtype)));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const auto& input_flat = input_tensor->flat<T>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("output", input_tensor->shape(),
                                            &output_tensor));
    auto output_flat = output_tensor->flat<int64_t>();

    functor::LaunchTensorToHashBucket<Device, T>()(
        context, num_buckets_, input_flat.data(), input_tensor->NumElements(),
        output_flat.data());
  }

 private:
  int64_t num_buckets_;

  TensorToHashBucketOp(const TensorToHashBucketOp&) = delete;
  void operator=(const TensorToHashBucketOp&) = delete;
};

#define REGISTER_CPU_KERNELS(type)                        \
  REGISTER_KERNEL_BUILDER(Name("_TensorToHashBucketFast") \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<type>("T"), \
                          TensorToHashBucketOp<CPUDevice, type>);

TF_CALL_INTEGRAL_TYPES(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA || MACHINA_USE_ROCM

#define REGISTER_GPU_KERNELS(type)                        \
  REGISTER_KERNEL_BUILDER(Name("_TensorToHashBucketFast") \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<type>("T"), \
                          TensorToHashBucketOp<GPUDevice, type>);

TF_CALL_INTEGRAL_TYPES(REGISTER_GPU_KERNELS);

#undef REGISTER_GPU_KERNELS

#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM

}  // namespace machina
