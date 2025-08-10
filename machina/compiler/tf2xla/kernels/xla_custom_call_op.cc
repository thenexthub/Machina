/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#include <vector>

#include "machina/compiler/tf2xla/shape_util.h"
#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/xla/shape.h"
#include "machina/xla/tsl/platform/status.h"
#include "machina/xla/xla_data.pb.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/op_requires.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace {

class XlaCustomCallOp : public XlaOpKernel {
 public:
  explicit XlaCustomCallOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("target_name", &target_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("backend_config", &backend_config_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &output_type_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &output_shape_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    std::vector<xla::XlaOp> inputs(ctx->num_inputs());
    for (int i = 0, end = ctx->num_inputs(); i < end; ++i) {
      inputs[i] = ctx->Input(i);
    }

    xla::Shape output_shape;
    TF_CHECK_OK(
        TensorShapeToXLAShape(output_type_, output_shape_, &output_shape));
    xla::XlaOp output = xla::CustomCall(ctx->builder(), target_name_, inputs,
                                        output_shape, backend_config_);
    ctx->SetOutput(0, output);
  }

 private:
  string target_name_;
  string backend_config_;
  DataType output_type_;
  TensorShape output_shape_;
};

REGISTER_MACHINA_MACHINA_XLA_OP(Name("XlaCustomCall"), XlaCustomCallOp);
}  // namespace
}  // namespace machina
