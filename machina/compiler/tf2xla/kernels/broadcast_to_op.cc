/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
#include <vector>

#include "machina/compiler/tf2xla/lib/broadcast.h"
#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace {

class BroadcastToOp : public XlaOpKernel {
 public:
  explicit BroadcastToOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    TensorShape output_shape;
    OP_REQUIRES_OK(context,
                   context->ConstantInputAsShape(
                       1, &output_shape, xla::ValueInferenceMode::kUpperBound));
    auto output_status_or =
        BroadcastTo(context->Input(0), output_shape.dim_sizes());
    OP_REQUIRES_OK(context, output_status_or.status());
    auto output = output_status_or.value();
    std::vector<bool> dynamic_dims;
    OP_REQUIRES_OK(
        context, context->ResolveInputDynamismIntoPredVector(1, &dynamic_dims));
    for (int64_t dim = 0; dim < dynamic_dims.size(); ++dim) {
      if (dynamic_dims[dim]) {
        output = xla::SetDimensionSize(
            output,
            xla::Reshape(xla::Slice(context->Input(1), {dim}, {dim + 1}, {1}),
                         {}),
            dim);
      }
    }

    context->SetOutput(0, output);
  }
};

REGISTER_MACHINA_MACHINA_XLA_OP(Name("BroadcastTo").CompileTimeConstantInput("shape"),
                BroadcastToOp);

}  // namespace
}  // namespace machina
