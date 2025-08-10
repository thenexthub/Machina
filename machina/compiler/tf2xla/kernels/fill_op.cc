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

// XLA-specific Fill Op.

#include <cstdint>
#include <vector>

#include "machina/compiler/tf2xla/type_util.h"
#include "machina/compiler/tf2xla/xla_helpers.h"
#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/hlo/builder/value_inference.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/xla/xla_data.pb.h"
#include "machina/core/framework/kernel_def_builder.h"
#include "machina/core/framework/register_types.h"
#include "machina/core/framework/tensor_shape.h"

namespace machina {
namespace {

class FillOp : public XlaOpKernel {
 public:
  explicit FillOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    // The output of this Op is a tensor of shape 'dims_shape' with each
    // element set to the scalar 'dims_literal'.
    const TensorShape dims_shape = ctx->InputShape("dims");
    const TensorShape value_shape = ctx->InputShape("value");
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(dims_shape),
        errors::InvalidArgument("dims must be a vector of int32, got shape ",
                                dims_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(value_shape),
                errors::InvalidArgument("value must be a scalar, got shape ",
                                        value_shape.DebugString()));

    std::vector<int64_t> dims;
    OP_REQUIRES_OK(ctx,
                   ctx->ConstantInputAsIntVector(
                       "dims", &dims, xla::ValueInferenceMode::kUpperBound));
    std::vector<bool> dynamic_dims;
    OP_REQUIRES_OK(
        ctx, ctx->ResolveInputDynamismIntoPredVector("dims", &dynamic_dims));

    auto output = xla::Broadcast(ctx->Input("value"), dims);
    for (int64_t i = 0; i < dims.size(); ++i) {
      // If a dimension is dynamic, call set-dimension-size on the output.
      if (dynamic_dims[i]) {
        auto dynamic_dim_size = xla::Slice(ctx->Input(0), {i}, {i + 1}, {1});
        dynamic_dim_size = xla::Reshape(dynamic_dim_size, {});
        dynamic_dim_size = xla::ConvertElementType(dynamic_dim_size, xla::S32);
        output = xla::SetDimensionSize(output, dynamic_dim_size, i);
      }
    }
    ctx->SetOutput(0, output);
  }
};

REGISTER_MACHINA_MACHINA_XLA_OP(Name("Fill").CompileTimeConstantInput("dims"), FillOp);

}  // namespace
}  // namespace machina
