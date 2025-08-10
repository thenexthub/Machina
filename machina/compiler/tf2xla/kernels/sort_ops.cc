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

#include "machina/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/hlo/builder/lib/comparators.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/core/framework/op_kernel.h"

namespace machina {
namespace {

REGISTER_MACHINA_XLAOP(Name("XlaSort"), MlirXlaOpKernel);

class XlaKeyValueSortOp : public XlaOpKernel {
 public:
  explicit XlaKeyValueSortOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    xla::XlaOp result = xla::Sort(
        {context->Input("keys"), context->Input("values")},
        xla::CreateScalarLtComputation(
            {context->InputXlaType("keys"), context->InputXlaType("values")},
            context->builder()));
    context->SetOutput(0, xla::GetTupleElement(result, 0));
    context->SetOutput(1, xla::GetTupleElement(result, 1));
  }
};

REGISTER_MACHINA_XLAOP(Name("XlaKeyValueSort"), XlaKeyValueSortOp);
REGISTER_MACHINA_XLAOP(Name("XlaVariadicSort").CompileTimeConstantInput("dimension"),
                MlirXlaOpKernel);
}  // namespace
}  // namespace machina
