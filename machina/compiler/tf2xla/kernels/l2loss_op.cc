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

#include <cstdint>
#include <numeric>
#include <vector>

#include "machina/compiler/tf2xla/xla_helpers.h"
#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/types.pb.h"

namespace machina {
namespace {

class L2LossOp : public XlaOpKernel {
 public:
  explicit L2LossOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    std::vector<int64_t> dims(ctx->InputShape(0).dims());
    std::iota(dims.begin(), dims.end(), 0);

    DataType dtype = ctx->input_type(0);
    xla::XlaBuilder* const b = ctx->builder();

    //  output = sum(t ** 2) / 2
    const DataType accumulation_type = XlaHelpers::SumAccumulationType(dtype);
    auto t = XlaHelpers::ConvertElementType(ctx->Input(0), accumulation_type);
    auto square = xla::Mul(t, t);
    auto reduce = xla::Reduce(square, XlaHelpers::Zero(b, accumulation_type),
                              *ctx->GetOrCreateAdd(accumulation_type), dims);
    auto deconverted = XlaHelpers::ConvertElementType(reduce, dtype);
    auto two = XlaHelpers::IntegerLiteral(b, dtype, 2);
    ctx->SetOutput(0, xla::Div(deconverted, two));
  }
};

REGISTER_MACHINA_XLAOP(Name("L2Loss"), L2LossOp);

}  // namespace
}  // namespace machina
