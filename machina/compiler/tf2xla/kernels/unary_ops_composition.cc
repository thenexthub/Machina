/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include <functional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "machina/compiler/tf2xla/kernels/elu_op.h"
#include "machina/compiler/tf2xla/kernels/relu_op.h"
#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/hlo/builder/lib/constants.h"
#include "machina/xla/hlo/builder/lib/math.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/op_requires.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace {

using XlaUnaryOpGenerator = std::function<xla::XlaOp(xla::XlaOp)>;
using XlaOpGeneratorMap = absl::flat_hash_map<string, XlaUnaryOpGenerator>;

void PopulateXlaOpGeneratorMap(XlaOpGeneratorMap* op_generator_map) {
  auto add_xla_op_generator = [&](std::string name,
                                  XlaUnaryOpGenerator xla_op_generator) {
    CHECK(op_generator_map->insert({name, xla_op_generator}).second);
  };

#define ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Name) \
  add_xla_op_generator(#Name,      \
                       [](xla::XlaOp operand) { return xla::Name(operand); });

  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Abs);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Acos);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Acosh);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Asin);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Asinh);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Atan);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Atanh);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Ceil);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Cos);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Cosh);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Expm1);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Exp);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Floor);
  add_xla_op_generator(
      "Inv", [](xla::XlaOp x) { return xla::ScalarLike(x, 1.0) / x; });
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Log);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Log1p);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Neg);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Reciprocal);
  add_xla_op_generator("Rint", xla::RoundToEven);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Round);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Rsqrt);
  add_xla_op_generator("Sigmoid",
                       [](xla::XlaOp x) { return xla::Logistic(x); });
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Sin);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Sinh);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Sqrt);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Square);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Tan);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Tanh);

  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Elu);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Relu);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Relu6);
  ADD_MACHINA_MACHINA_XLA_OP_GENERATOR(Selu);

#undef ADD_MACHINA_MACHINA_XLA_OP_GENERATOR
}

const XlaOpGeneratorMap& GetXlaOpGeneratorMap() {
  static XlaOpGeneratorMap* result = []() {
    auto* result = new XlaOpGeneratorMap;
    PopulateXlaOpGeneratorMap(result);
    return result;
  }();

  return *result;
}

class UnaryOpsCompositionOp : public XlaOpKernel {
 public:
  explicit UnaryOpsCompositionOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("op_names", &op_names_));

    const XlaOpGeneratorMap& op_generator_map = GetXlaOpGeneratorMap();
    for (absl::string_view op_name : op_names_) {
      OP_REQUIRES(ctx, op_generator_map.contains(op_name),
                  errors::Unimplemented(
                      op_name, " not supported in _UnaryOpsComposition"));
    }
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp x = ctx->Input(0);
    const XlaOpGeneratorMap& op_generator_map = GetXlaOpGeneratorMap();
    for (absl::string_view op_name : op_names_) {
      x = op_generator_map.find(op_name)->second(x);
    }
    ctx->SetOutput(0, x);
  }

 private:
  std::vector<string> op_names_;
};

REGISTER_MACHINA_MACHINA_XLA_OP(Name("_UnaryOpsComposition"), UnaryOpsCompositionOp);

}  // namespace
}  // namespace machina
