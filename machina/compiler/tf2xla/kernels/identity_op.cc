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

#include "absl/log/check.h"
#include "machina/compiler/tf2xla/kernels/tensor_list_utils.h"
#include "machina/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/types.pb.h"

namespace machina {
namespace {

class IdentityOp : public XlaOpKernel {
 public:
  explicit IdentityOp(OpKernelConstruction* context) : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* ctx) override {
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      if (IsTensorListInput(ctx, i)) {
        ctx->SetTensorListOutput(i, ctx->Input(i));
      } else {
        DCHECK(ctx->input_type(i) != DT_VARIANT);
        // Forwards using the underlying op_kernel_context so both tensor and
        // resource values are forwarded correctly.
        ctx->op_kernel_context()->set_output(
            i, ctx->op_kernel_context()->input(i));
      }
    }
  }

 private:
  IdentityOp(const IdentityOp&) = delete;
  void operator=(const IdentityOp&) = delete;
};

// MACHINA_MACHINA_XLA_* devices also register a "real" Identity operator so we suppress the
// dummy operator using CompilationOnly().
REGISTER_MACHINA_MACHINA_XLA_OP(
    Name("Identity").AllowResourceTypes().AllowVariantTypes().CompilationOnly(),
    IdentityOp);
REGISTER_MACHINA_MACHINA_XLA_OP(Name("IdentityN")
                    .AllowResourceTypes()
                    .AllowVariantTypes()
                    .CompilationOnly(),
                IdentityOp);
REGISTER_MACHINA_MACHINA_XLA_OP(Name("PlaceholderWithDefault"), IdentityOp);
REGISTER_MACHINA_MACHINA_XLA_OP(Name("PreventGradient"), MlirXlaOpKernel);
REGISTER_MACHINA_MACHINA_XLA_OP(Name("StopGradient").AllowVariantTypes(), IdentityOp);
REGISTER_MACHINA_MACHINA_XLA_OP(Name("Snapshot"), IdentityOp);
REGISTER_MACHINA_MACHINA_XLA_OP(Name("_EagerConst"), IdentityOp);

}  // namespace
}  // namespace machina
