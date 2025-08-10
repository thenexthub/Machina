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

// XLA-specific ensure_shape Op.

#include "absl/log/log.h"
#include "machina/compiler/tf2xla/type_util.h"
#include "machina/compiler/tf2xla/xla_helpers.h"
#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/xla/literal.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/register_types.h"
#include "machina/core/framework/tensor.h"

namespace machina {
namespace {

class EnsureShapeOp : public XlaOpKernel {
 public:
  explicit EnsureShapeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &expected_shape_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape shape = ctx->InputShape(0);

    // valiate shape
    OP_REQUIRES(
        ctx, expected_shape_.IsCompatibleWith(shape),
        errors::InvalidArgument("Shape of tensor ", this->def().input(0), " ",
                                shape.DebugString(),
                                " is not compatible with expected shape ",
                                expected_shape_.DebugString(), "."));

    // If the shape dimension in `expected_shape_` is already static, we would
    // remove the dynamic dimensions in XLA dynamic padder. Here we don't check
    // whether the original input has dynamic shapes, because
    // `ctx->ResolveInputDynamismIntoPredVector` runs a DFS underneath which is
    // more expensive.
    xla::XlaOp tensor = ctx->Input(0);
    for (int i = 0; i < expected_shape_.dims(); ++i) {
      if (expected_shape_.dim_size(i) > 0) {
        VLOG(1) << "RemoveDynamicDimension: " << i << " of shape "
                << shape.DebugString();
        tensor = xla::RemoveDynamicDimension(tensor, i);
      }
    }

    // If shape matches, outputs the tensor.
    ctx->SetOutput(0, tensor);
  }

 private:
  PartialTensorShape expected_shape_;
};

REGISTER_MACHINA_MACHINA_XLA_OP(Name("EnsureShape"), EnsureShapeOp);

}  // namespace
}  // namespace machina
