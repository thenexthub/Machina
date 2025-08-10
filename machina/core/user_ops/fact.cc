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

#include <string>

#include "machina/core/framework/common_shape_fns.h"
#include "machina/core/framework/op.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/op_requires.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.h"
#include "machina/core/platform/tstring.h"

REGISTER_OP("Fact")
    .Output("fact: string")
    .SetShapeFn(machina::shape_inference::UnknownShape);

class FactOp : public machina::OpKernel {
 public:
  explicit FactOp(machina::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(machina::OpKernelContext* context) override {
    // Output a scalar string.
    machina::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, machina::TensorShape(), &output_tensor));
    using machina::string;
    auto output = output_tensor->template scalar<machina::tstring>();

    output() = "0! == 1";
  }
};

REGISTER_KERNEL_BUILDER(Name("Fact").Device(machina::DEVICE_CPU), FactOp);
