/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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
#include "machina/core/framework/op.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/shape_inference.h"

namespace machina {

REGISTER_OP("Double")
    .Input("input: T")
    .Output("doubled: T")
    .Attr("T: {int32, float}")
    .SetShapeFn([](::machina::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return absl::OkStatus();
    });

template <typename T>
class DoubleOp : public OpKernel {
 public:
  explicit DoubleOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input_flat = input_tensor.flat<T>();

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<T>();

    // Set all but the first element of the output tensor to 0.
    const int N = input_flat.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = 2 * input_flat(i);
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("Double").Device(DEVICE_CPU).TypeConstraint<int32>("T"),
    DoubleOp<int32>);
REGISTER_KERNEL_BUILDER(
    Name("Double").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    DoubleOp<float>);
}  // namespace machina
