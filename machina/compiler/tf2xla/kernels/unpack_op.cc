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

// XLA Unpack operator.

#include <cstdint>
#include <vector>

#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/op_requires.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/platform/errors.h"

namespace machina {
namespace {

class UnpackOp : public XlaOpKernel {
 public:
  explicit UnpackOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const int num = num_outputs();
    const TensorShape input_shape = ctx->InputShape(0);

    int axis = axis_;
    if (axis < 0) axis += input_shape.dims();

    OP_REQUIRES(ctx, 0 <= axis && axis < input_shape.dims(),
                errors::InvalidArgument("axis = ", axis_, " not in [",
                                        -input_shape.dims(), ", ",
                                        input_shape.dims(), ")"));

    OP_REQUIRES(
        ctx, input_shape.dims() > 0 && input_shape.dim_size(axis) == num,
        errors::InvalidArgument("Input shape axis ", axis, " must equal ", num,
                                ", got shape ", input_shape.DebugString()));

    auto output_shape = input_shape;
    output_shape.RemoveDim(axis);

    auto input = ctx->Input(0);

    std::vector<int64_t> start_indices(input_shape.dims(), 0);
    std::vector<int64_t> limit_indices(input_shape.dims());
    std::vector<int64_t> strides(input_shape.dims(), 1);
    for (int i = 0; i < input_shape.dims(); ++i) {
      limit_indices[i] = input_shape.dim_size(i);
    }

    for (int i = 0; i < num; ++i) {
      start_indices[axis] = i;
      limit_indices[axis] = i + 1;
      auto slice = xla::Slice(input, start_indices, limit_indices, strides);
      // Reshape to drop the 'axis' dimension.
      auto result = xla::Reshape(slice, output_shape.dim_sizes());
      ctx->SetOutput(i, result);
    }
  }

 private:
  int axis_;
};

REGISTER_MACHINA_MACHINA_XLA_OP(Name("Unpack"), UnpackOp);

}  // namespace
}  // namespace machina
