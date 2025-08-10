/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

// XLA-specific Empty Op.

#include <cstdint>
#include <vector>

#include "machina/compiler/tf2xla/type_util.h"
#include "machina/compiler/tf2xla/xla_helpers.h"
#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/hlo/builder/lib/constants.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/xla/xla_data.pb.h"
#include "machina/core/framework/kernel_def_builder.h"
#include "machina/core/framework/register_types.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.pb.h"

namespace machina {
namespace {

class EmptyOp : public XlaOpKernel {
 public:
  explicit EmptyOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(dtype_, &type_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("init", &init_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    // The output of this Op is a tensor of shape 'shape' with each
    // element set to the default value of 'dtype'. If 'init' is false then
    // the result values may be left undefined, though we don't do that here.
    const TensorShape shape_shape = ctx->InputShape("shape");
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(shape_shape),
        errors::InvalidArgument("shape must be a vector of int32, got shape ",
                                shape_shape.DebugString()));

    std::vector<int64_t> shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector("shape", &shape));

    auto default_value = xla::Zero(ctx->builder(), type_);
    auto result = xla::Broadcast(default_value, shape);
    ctx->SetOutput(0, result);
  }

 private:
  DataType dtype_;
  xla::PrimitiveType type_;
  bool init_;
};

REGISTER_MACHINA_MACHINA_XLA_OP(Name("Empty").CompileTimeConstantInput("shape"), EmptyOp);

}  // namespace
}  // namespace machina
