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

#include "machina/compiler/tf2xla/shape_util.h"
#include "machina/compiler/tf2xla/xla_compiler.h"
#include "machina/compiler/tf2xla/xla_helpers.h"
#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/hlo/builder/lib/constants.h"
#include "machina/core/framework/kernel_def_builder.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.pb.h"

namespace machina {

// This OpKernel implements the FakeParam Op for XLA JIT devices. Create zeros
// with the appropriate shape for FakeParam op.
class XlaFakeParamOp : public XlaOpKernel {
 public:
  explicit XlaFakeParamOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    DataType dtype;
    // Tensor shape can be unknown.
    PartialTensorShape tensor_shape;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &tensor_shape));
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, tensor_shape, &shape_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    ctx->SetOutput(0, xla::Zeros(b, shape_));
  }

 private:
  xla::Shape shape_;

  XlaFakeParamOp(const XlaFakeParamOp&) = delete;
  void operator=(const XlaFakeParamOp&) = delete;
};

REGISTER_MACHINA_XLAOP(Name("FakeParam"), XlaFakeParamOp);

}  // namespace machina
