/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

using namespace machina;  // NOLINT(build/namespaces)

REGISTER_OP("AddOne")
    .Input("input: int32")
    .Output("output: int32")
    .Doc(R"doc(
Adds 1 to all elements of the tensor.

output: A Tensor.
  output = input + 1
)doc");

void AddOneKernelLauncher(const int* in, int N, int* out);

class AddOneOp : public OpKernel {
 public:
  explicit AddOneOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32_t>();

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template flat<int32_t>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    // Call the cuda kernel launcher
    AddOneKernelLauncher(input.data(), N, output.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("AddOne").Device(DEVICE_GPU), AddOneOp);
