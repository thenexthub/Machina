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

// XLA Pack operator.

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

class PackOp : public XlaOpKernel {
 public:
  explicit PackOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    std::vector<xla::XlaOp> values;
    std::vector<TensorShape> shapes;
    OP_REQUIRES_OK(ctx, ctx->InputList("values", &values, &shapes));
    const int num = values.size();

    OP_REQUIRES(ctx, num >= 0,
                errors::InvalidArgument("Pack requires >= 1 arguments"));

    // Verify that all input shapes match
    for (int i = 1; i < num; i++) {
      OP_REQUIRES(ctx, shapes[0].IsSameSize(shapes[i]),
                  errors::InvalidArgument(
                      "Shapes of all inputs must match: values[0].shape = ",
                      shapes[0].DebugString(), " != values[", i, "].shape = ",
                      shapes[i].DebugString()));
    }

    int expanded_num_dims = shapes[0].dims() + 1;
    int axis = axis_;
    if (axis < 0) axis += expanded_num_dims;

    OP_REQUIRES(ctx, 0 <= axis && axis < expanded_num_dims,
                errors::InvalidArgument("axis = ", axis_, " not in [",
                                        -expanded_num_dims, ", ",
                                        expanded_num_dims, ")"));

    std::vector<xla::XlaOp> reshaped_inputs(num);

    TensorShape child_shape(shapes[0]);
    child_shape.InsertDim(axis, 1);

    for (int i = 0; i < num; ++i) {
      // Reshape the inputs to have an extra dimension of size 1.
      reshaped_inputs[i] = xla::Reshape(values[i], child_shape.dim_sizes());
    }

    ctx->SetOutput(0, xla::ConcatInDim(ctx->builder(), reshaped_inputs, axis));
  }

 private:
  int axis_;
};

REGISTER_MACHINA_MACHINA_XLA_OP(Name("Pack"), PackOp);

}  // namespace
}  // namespace machina
