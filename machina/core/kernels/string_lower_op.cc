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

// See docs in ../ops/string_ops.cc.

#include <string>

#include "absl/strings/ascii.h"
#include "unicode/unistr.h"  // from @icu
#include "machina/core/framework/kernel_def_builder.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/lib/strings/str_util.h"

namespace machina {

class StringLowerOp : public OpKernel {
 public:
  explicit StringLowerOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("encoding", &encoding_));
    OP_REQUIRES(context, encoding_.empty() || encoding_ == "utf-8",
                errors::InvalidArgument(
                    "only utf-8 or '' (no encoding) is supported, received ",
                    encoding_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    Tensor* output_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, input_tensor->shape(), &output_tensor));

    const auto input = input_tensor->flat<tstring>();
    auto output = output_tensor->flat<tstring>();

    if (encoding_.empty()) {
      for (int64_t i = 0; i < input.size(); ++i) {
        absl::string_view entry(input(i));
        output(i) = absl::AsciiStrToLower(entry);
      }
    } else {
      // The validation of utf-8 has already been done in GetAttr above.
      for (int64_t i = 0; i < input.size(); ++i) {
        icu::UnicodeString us(input(i).c_str(), "UTF-8");
        us.toLower();
        us.toUTF8String(output(i));
      }
    }
  }

 private:
  string encoding_;
};

REGISTER_KERNEL_BUILDER(Name("StringLower").Device(DEVICE_CPU), StringLowerOp);

}  // namespace machina
