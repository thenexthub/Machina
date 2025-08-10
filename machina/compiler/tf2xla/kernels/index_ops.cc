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

// Native XLA implementations of indexing ops.

#include "machina/compiler/tf2xla/kernels/index_ops.h"

#include <cstdint>

#include "machina/compiler/tf2xla/type_util.h"
#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/hlo/builder/lib/arithmetic.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/xla/xla_data.pb.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/op_requires.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/errors.h"

namespace machina {
XlaArgMinMaxOp::XlaArgMinMaxOp(OpKernelConstruction* ctx, bool is_min)
    : XlaOpKernel(ctx), is_min_(is_min) {}

void XlaArgMinMaxOp::Compile(XlaOpKernelContext* ctx) {
  const TensorShape input_shape = ctx->InputShape(0);
  const TensorShape dimension_shape = ctx->InputShape(1);

  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(dimension_shape),
              errors::InvalidArgument(
                  "dim must be a scalar, but received tensor of shape: ",
                  dimension_shape.DebugString()));

  int64_t dim;
  OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(1, &dim));

  const int input_dims = input_shape.dims();
  const int axis = dim < 0 ? dim + input_dims : dim;

  OP_REQUIRES(
      ctx, axis >= 0 && axis < input_dims,
      errors::InvalidArgument("Expected dimension in the range [", -input_dims,
                              ", ", input_dims, "), but got ", dim));
  const int64_t axis_size = input_shape.dim_size(axis);
  OP_REQUIRES(
      ctx, axis_size > 0,
      errors::InvalidArgument("Reduction axis ", dim, " is empty in shape ",
                              input_shape.DebugString()));

  DataType index_type = output_type(0);
  xla::PrimitiveType index_xla_type;
  OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(index_type, &index_xla_type));

  xla::XlaOp input = ctx->Input(0);
  xla::XlaOp output =
      xla::ArgMinMax(input, index_xla_type, axis, /*is_min=*/is_min_);

  ctx->SetOutput(0, output);
}

XlaArgMaxOp::XlaArgMaxOp(OpKernelConstruction* ctx)
    : XlaArgMinMaxOp(ctx, /*is_min=*/false) {}
REGISTER_MACHINA_MACHINA_XLA_OP(Name("ArgMax").CompileTimeConstantInput("dimension"),
                XlaArgMaxOp);

namespace {

class XlaArgMinOp : public XlaArgMinMaxOp {
 public:
  explicit XlaArgMinOp(OpKernelConstruction* ctx);
};
XlaArgMinOp::XlaArgMinOp(OpKernelConstruction* ctx)
    : XlaArgMinMaxOp(ctx, /*is_min=*/true) {}
REGISTER_MACHINA_MACHINA_XLA_OP(Name("ArgMin").CompileTimeConstantInput("dimension"),
                XlaArgMinOp);

}  // namespace
}  // namespace machina
