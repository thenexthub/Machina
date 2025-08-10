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

// An example Op.

#include "machina/core/framework/op.h"
#include "machina/core/framework/op_kernel.h"

namespace machina {

REGISTER_OP("Ackermann")
    .Output("ackermann: string")
    .Doc(R"doc(
Output a fact about the ackermann function.
)doc");

class AckermannOp : public OpKernel {
 public:
  explicit AckermannOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Output a scalar string.
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape(), &output_tensor));
    auto output = output_tensor->scalar<tstring>();

    output() = "A(m, 0) == A(m-1, 1)";
  }
};

REGISTER_KERNEL_BUILDER(Name("Ackermann").Device(DEVICE_CPU), AckermannOp);

}  // namespace machina
