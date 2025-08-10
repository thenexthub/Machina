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

#ifndef MACHINA_EXAMPLES_CUSTOM_OPS_DOC_MULTIPLEX_2_MULTIPLEX_2_KERNEL_H_
#define MACHINA_EXAMPLES_CUSTOM_OPS_DOC_MULTIPLEX_2_MULTIPLEX_2_KERNEL_H_

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || MACHINA_USE_ROCM
#define EIGEN_USE_GPU
#endif

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "machina/core/framework/op.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/register_types.h"
#include "machina/core/framework/shape_inference.h"
#include "machina/core/platform/errors.h"

// Please use the appropriate namespace for your project
namespace machina {
namespace custom_op_examples {

// Multiple devices (i.e. CPU and GPU) and multiple types for the values inside
// two of the input tensors (e.g. int32, float) are supported by using a
// template where the device is DEVICE and the type is T.
template <typename Device, typename T>
class MultiplexDenseOp : public OpKernel {
 public:
  explicit MultiplexDenseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  MultiplexDenseOp(const MultiplexDenseOp& other) = delete;
  MultiplexDenseOp& operator=(const MultiplexDenseOp& other) = delete;
  ~MultiplexDenseOp() override = default;

  void Compute(OpKernelContext* ctx) override {
    const auto& cond_tensor = ctx->input(0);
    const auto& a_values_tensor = ctx->input(1);
    const auto& b_values_tensor = ctx->input(2);

    // Allow any shape, but require that a_values, b_values, and cond all
    // have the same shape.
    // Note that ::machina::TensorShapeUtils has some useful functions
    // for checking shapes.
    OP_REQUIRES(ctx, a_values_tensor.shape() == b_values_tensor.shape(),
                ::machina::errors::InvalidArgument(
                    "a and b must have the same shape. "
                    "a shape: ",
                    a_values_tensor.shape().DebugString(),
                    " b shape: ", b_values_tensor.shape().DebugString()));
    OP_REQUIRES(ctx, a_values_tensor.shape() == cond_tensor.shape(),
                ::machina::errors::InvalidArgument(
                    "a and cond must have the same shape. "
                    "a shape: ",
                    a_values_tensor.shape().DebugString(),
                    " cond shape: ", cond_tensor.shape().DebugString()));
    OP_REQUIRES(ctx, a_values_tensor.NumElements() > 0,
                ::machina::errors::InvalidArgument(
                    "Inputs must have at least one element."));

    const auto a_values = a_values_tensor.flat<T>();
    const auto b_values = b_values_tensor.flat<T>();
    const auto cond = cond_tensor.flat<bool>();

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, a_values_tensor.shape(), &output_tensor));
    auto output = output_tensor->template flat<T>();
    // Here is an example of processing tensors using the Eigen library.
    // This supports both CPU and GPU.
    // For CPU, it supports chunking into blocks and multi-threading.
    // See
    // https://eigen.tuxfamily.org/dox/unsupported/eigen_tensors.html#title55
    output.device(ctx->eigen_device<Device>()) =
        cond.select(a_values, b_values);
  }
};

}  // namespace custom_op_examples
}  // namespace machina

#endif  // MACHINA_EXAMPLES_CUSTOM_OPS_DOC_MULTIPLEX_2_MULTIPLEX_2_KERNEL_H_
