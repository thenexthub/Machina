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

#include "machina/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_helpers.h"
#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/util/tensor_format.h"

namespace machina {
namespace {

class BiasOp : public XlaOpKernel {
 public:
  explicit BiasOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    string data_format;
    if (ctx->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(ctx, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      data_format_ = FORMAT_NHWC;
    }
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    const TensorShape bias_shape = ctx->InputShape(1);

    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrixOrHigher(input_shape),
                errors::InvalidArgument("Input tensor must be at least 2D: ",
                                        input_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(bias_shape),
                errors::InvalidArgument("Biases must be 1D: ",
                                        bias_shape.DebugString()));

    // feature_dim is the channel (C) dimension of the data.
    int feature_dim = (data_format_ == FORMAT_NHWC)
                          ? input_shape.dims() - 1
                          : /*data_format == FORMAT_NCHW*/ 1;
    OP_REQUIRES(
        ctx, feature_dim >= 0,
        errors::InvalidArgument("Input tensor does not have enough dimensions "
                                "to contain the feature dimension"));
    OP_REQUIRES(
        ctx, bias_shape.dim_size(0) == input_shape.dim_size(feature_dim),
        errors::InvalidArgument(
            "Must provide as many biases as the last dimension "
            "of the input tensor: ",
            bias_shape.DebugString(), " vs. ", input_shape.DebugString()));

    xla::XlaOp result = xla::Add(ctx->Input(0), ctx->Input(1), {feature_dim});
    ctx->SetOutput(0, result);
  }

 private:
  TensorFormat data_format_;
};

REGISTER_MACHINA_MACHINA_XLA_OP(Name("BiasAdd"), BiasOp);
REGISTER_MACHINA_MACHINA_XLA_OP(Name("BiasAddV1"), BiasOp);
REGISTER_MACHINA_MACHINA_XLA_OP(Name("BiasAddGrad"), MlirXlaOpKernel);

}  // namespace
}  // namespace machina
