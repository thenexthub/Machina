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

#include "machina/compiler/tf2xla/xla_context.h"
#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/op_requires.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/types.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/errors.h"

namespace machina {
namespace {

// This TensorFlow op indicates that its input should be treated as a
// specific return value from a function.
class RetvalOp : public XlaOpKernel {
 public:
  explicit RetvalOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("index", &index_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const Tensor& input = ctx->op_kernel_context()->input(0);

    // Types that cannot be copied using memcpy (like DT_VARIANT types that
    // represent Tensor Lists) are wrapped in a DT_UINT8 and hence the type
    // mismatches. Skip the test in such cases. See
    // XlaOpKernelContext::SetOutputExpression for details.
    if (DataTypeCanUseMemcpy(dtype_)) {
      OP_REQUIRES(ctx, input.dtype() == dtype_,
                  errors::InvalidArgument(
                      "Type mismatch: actual ", DataTypeString(input.dtype()),
                      " vs. expect ", DataTypeString(dtype_)));
    }
    auto frame = ctx->call_frame();
    if (frame) {
      // If 'frame' is non-null, this is an inner function call inside a JIT
      // compilation.
      OP_REQUIRES_OK(ctx, frame->SetRetval(index_, input));
    } else {
      ctx->xla_context()->SetRetval(index_, ctx->InputExpression(0));
    }
  }

 private:
  // The index of this return value in the returned tuple.
  int index_;
  DataType dtype_;

  RetvalOp(const RetvalOp&) = delete;
  void operator=(const RetvalOp&) = delete;
};

REGISTER_MACHINA_MACHINA_XLA_OP(
    Name("_Retval").AllowResourceTypes().AllowVariantTypes().CompilationOnly(),
    RetvalOp);

}  // anonymous namespace
}  // namespace machina
