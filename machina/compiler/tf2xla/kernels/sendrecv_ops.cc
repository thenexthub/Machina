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
#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/xla/shape.h"
#include "machina/xla/xla_data.pb.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/op_requires.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace {

class SendOp : public XlaOpKernel {
 public:
  explicit SendOp(OpKernelConstruction* ctx);
  void Compile(XlaOpKernelContext* ctx) override;

 private:
  string tensor_name_;

  SendOp(const SendOp&) = delete;
  void operator=(const SendOp&) = delete;
};

SendOp::SendOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name_));
}

void SendOp::Compile(XlaOpKernelContext* ctx) {
  XlaCompiler* compiler = ctx->compiler();
  xla::ChannelHandle channel;
  OP_REQUIRES_OK(ctx, compiler->GetChannelHandle(tensor_name_, &channel));
  xla::Send(ctx->Input(0), channel);
}

REGISTER_MACHINA_MACHINA_XLA_OP(Name("XlaSend"), SendOp);

class RecvOp : public XlaOpKernel {
 public:
  explicit RecvOp(OpKernelConstruction* ctx);
  void Compile(XlaOpKernelContext* ctx) override;

 private:
  string tensor_name_;
  xla::Shape shape_;

  RecvOp(const RecvOp&) = delete;
  void operator=(const RecvOp&) = delete;
};

RecvOp::RecvOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name_));

  TensorShape tensor_shape;
  DataType dtype;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &tensor_shape));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype));
  OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, tensor_shape, &shape_));
}

void RecvOp::Compile(XlaOpKernelContext* ctx) {
  XlaCompiler* compiler = ctx->compiler();
  xla::ChannelHandle channel;
  OP_REQUIRES_OK(ctx, compiler->GetChannelHandle(tensor_name_, &channel));
  ctx->SetOutput(0, xla::Recv(ctx->builder(), shape_, channel));
}

REGISTER_MACHINA_MACHINA_XLA_OP(Name("XlaRecv"), RecvOp);

}  // namespace
}  // namespace machina
