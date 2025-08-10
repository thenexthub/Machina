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

#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/hlo/builder/lib/quantize.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/xla/xla_data.pb.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/op_requires.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace {

class XlaDequantizeOp : public XlaOpKernel {
 public:
  explicit XlaDequantizeOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("min_range", &min_range_));
    OP_REQUIRES_OK(context, context->GetAttr("max_range", &max_range_));
    OP_REQUIRES_OK(context, context->GetAttr("mode", &mode_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("transpose_output", &transpose_output_));
  }

  void Compile(XlaOpKernelContext* context) override {
    const xla::XlaOp& input = context->Input(0);

    xla::QuantizedRange range(min_range_, max_range_);

    xla::XlaOp output =
        xla::Dequantize<uint8>(input, range, mode_, transpose_output_);
    context->SetOutput(0, output);
  }

 private:
  float min_range_;
  float max_range_;
  bool transpose_output_;
  string mode_;
  XlaDequantizeOp(const XlaDequantizeOp&) = delete;
  void operator=(const XlaDequantizeOp&) = delete;
};

REGISTER_MACHINA_MACHINA_XLA_OP(Name("XlaDequantize"), XlaDequantizeOp);

}  // namespace
}  // namespace machina
