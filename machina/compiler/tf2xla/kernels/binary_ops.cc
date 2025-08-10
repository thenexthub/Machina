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

// Native XLA implementations of simple binary Ops

#include <cstdint>
#include <tuple>
#include <vector>

#include "absl/types/span.h"
#include "machina/compiler/tf2xla/kernels/cwise_ops.h"
#include "machina/compiler/tf2xla/lib/broadcast.h"
#include "machina/compiler/tf2xla/shape_util.h"
#include "machina/compiler/tf2xla/xla_helpers.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/client/client_library.h"
#include "machina/xla/hlo/builder/lib/constants.h"
#include "machina/xla/hlo/builder/lib/math.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/xla/primitive_util.h"
#include "machina/xla/xla_data.pb.h"
#include "machina/core/framework/kernel_def_builder.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/types.h"
#include "machina/core/framework/types.pb.h"

namespace machina {
namespace {

// A subclass of a XlaBinaryOp must build the computation that
// describes the (tensor,tensor)->tensor function to apply to each element of
// the input.
#define MACHINA_MACHINA_XLA_MAKE_BINARY(NAME, HLO)                                         \
  class NAME##Op : public XlaBinaryOp {                                    \
   public:                                                                 \
    explicit NAME##Op(OpKernelConstruction* ctx) : XlaBinaryOp(ctx) {}     \
    xla::XlaOp Computation(                                                \
        XlaOpKernelContext* ctx, const xla::XlaOp& lhs,                    \
        const absl::Span<const int64_t>& lhs_shape, const xla::XlaOp& rhs, \
        const absl::Span<const int64_t>& rhs_shape,                        \
        const BCast& broadcast_helper,                                     \
        const std::vector<int64_t>& extend_dimensions) override {          \
      xla::XlaBuilder* b = ctx->builder();                                 \
      (void)b;                                                             \
      (void)lhs_shape;                                                     \
      (void)rhs_shape;                                                     \
      (void)extend_dimensions;                                             \
      return HLO;                                                          \
    }                                                                      \
  };                                                                       \
  REGISTER_MACHINA_MACHINA_XLA_OP(Name(#NAME), NAME##Op)

MACHINA_MACHINA_XLA_MAKE_BINARY(Add, xla::Add(lhs, rhs, extend_dimensions));
MACHINA_MACHINA_XLA_MAKE_BINARY(AddV2, xla::Add(lhs, rhs, extend_dimensions));
MACHINA_MACHINA_XLA_MAKE_BINARY(Sub, xla::Sub(lhs, rhs, extend_dimensions));
MACHINA_MACHINA_XLA_MAKE_BINARY(Mul, xla::Mul(lhs, rhs, extend_dimensions));
MACHINA_MACHINA_XLA_MAKE_BINARY(Div, xla::Div(lhs, rhs, extend_dimensions));

MACHINA_MACHINA_XLA_MAKE_BINARY(Atan2, xla::Atan2(lhs, rhs, extend_dimensions));
MACHINA_MACHINA_XLA_MAKE_BINARY(Complex, xla::Complex(lhs, rhs, extend_dimensions));

// Implementation of DivNoNan. Pseudo-code:
// if (y == 0) {
//   return 0
// } else {
//   return x / y;
// }
static xla::XlaOp DivNoNanImpl(xla::XlaBuilder* b, DataType dtype, xla::XlaOp x,
                               xla::XlaOp y, const BCast& broadcast_helper) {
  std::tie(x, y) = XlaBinaryOp::Broadcast(x, y, broadcast_helper);
  auto zero = XlaHelpers::Zero(b, dtype);
  auto y_equals_0 = xla::Eq(y, zero);
  auto zeros = xla::ZerosLike(x);
  auto result = xla::Select(y_equals_0, zeros, xla::Div(x, y));
  return result;
}
MACHINA_MACHINA_XLA_MAKE_BINARY(DivNoNan,
                DivNoNanImpl(b, input_type(0), lhs, rhs, broadcast_helper));

// Implementation of MulNoNan. Pseudo-code:
// if (y == 0) {
//   return 0
// } else {
//   return x * y;
// }
static xla::XlaOp MulNoNanImpl(xla::XlaBuilder* b, DataType dtype, xla::XlaOp x,
                               xla::XlaOp y, const BCast& broadcast_helper) {
  std::tie(x, y) = XlaBinaryOp::Broadcast(x, y, broadcast_helper);
  auto zero = XlaHelpers::Zero(b, dtype);
  auto y_equals_0 = xla::Eq(y, zero);
  auto zeros = xla::ZerosLike(x);
  auto result = xla::Select(y_equals_0, zeros, xla::Mul(x, y));
  return result;
}
MACHINA_MACHINA_XLA_MAKE_BINARY(MulNoNan,
                MulNoNanImpl(b, input_type(0), lhs, rhs, broadcast_helper));

// Implementation of FloorDiv.
//
// For floating-point values, simply returns floor(x / y).  For integers, does:
//
// z = x / y
// if (z * y != x && (x < 0) != (y < 0)) {
//   return  z - 1;
// } else {
//   return z;
// }
static xla::XlaOp FloorDivImpl(xla::XlaBuilder* b, DataType dtype, xla::XlaOp x,
                               xla::XlaOp y, const BCast& broadcast_helper) {
  std::tie(x, y) = XlaBinaryOp::Broadcast(x, y, broadcast_helper);
  if (DataTypeIsFloating(dtype)) {
    if (dtype == DataType::DT_BFLOAT16) {
      // The result of a BF16 division may produce the Ceil of what was
      // computed by F32 division, so avoid end user confusion by doing the
      // intermediate divide in F32.
      return xla::ConvertElementType(
          xla::Floor(xla::Div(xla::ConvertElementType(x, xla::F32),
                              xla::ConvertElementType(y, xla::F32))),
          xla::BF16);
    } else {
      return xla::Floor(xla::Div(x, y));
    }
  }
  if (DataTypeIsUnsigned(dtype)) {
    return xla::Div(x, y);
  }
  auto zero = XlaHelpers::Zero(b, dtype);
  auto one = XlaHelpers::One(b, dtype);
  auto x_div_y = xla::Div(x, y);
  auto round_down = xla::And(xla::Ne(xla::Mul(x_div_y, y), x),
                             xla::Ne(xla::Lt(x, zero), xla::Lt(y, zero)));
  return xla::Select(round_down, xla::Sub(x_div_y, one), x_div_y);
}
MACHINA_MACHINA_XLA_MAKE_BINARY(FloorDiv,
                FloorDivImpl(b, input_type(0), lhs, rhs, broadcast_helper));

xla::XlaOp XlogyImpl(xla::XlaOp x, xla::XlaOp y,
                     const BCast& broadcast_helper) {
  std::tie(x, y) = XlaBinaryOp::Broadcast(x, y, broadcast_helper);
  auto zero = xla::ZerosLike(x);
  auto is_zero = xla::Eq(x, zero);
  return xla::Select(is_zero, zero, xla::Mul(x, xla::Log(y)));
}
MACHINA_MACHINA_XLA_MAKE_BINARY(Xlogy, XlogyImpl(lhs, rhs, broadcast_helper));

xla::XlaOp Xlog1pyImpl(xla::XlaOp x, xla::XlaOp y,
                       const BCast& broadcast_helper) {
  std::tie(x, y) = XlaBinaryOp::Broadcast(x, y, broadcast_helper);
  auto non_zero = xla::Mul(x, xla::Log1p(y));
  auto zero = xla::ZerosLike(non_zero);
  auto x_is_zero = xla::Eq(x, zero);
  return xla::Select(x_is_zero, zero, non_zero);
}
MACHINA_MACHINA_XLA_MAKE_BINARY(Xlog1py, Xlog1pyImpl(lhs, rhs, broadcast_helper));

xla::XlaOp XdivyImpl(xla::XlaOp x, xla::XlaOp y,
                     const BCast& broadcast_helper) {
  std::tie(x, y) = XlaBinaryOp::Broadcast(x, y, broadcast_helper);
  auto zero = xla::ZerosLike(x);
  auto is_zero = xla::Eq(x, zero);
  return xla::Select(is_zero, zero, xla::Div(x, y));
}
MACHINA_MACHINA_XLA_MAKE_BINARY(Xdivy, XdivyImpl(lhs, rhs, broadcast_helper));

// Implementation of FloorMod. Pseudo-code:
// T trunc_mod = std::fmod(x, y);
// return trunc_mod != 0 && (y < 0 != trunc_mod < 0) ? trunc_mod + y
//                                                   : trunc_mod;
static xla::XlaOp FloorModImpl(xla::XlaBuilder* b, DataType dtype, xla::XlaOp x,
                               xla::XlaOp y, const BCast& broadcast_helper) {
  std::tie(x, y) = XlaBinaryOp::Broadcast(x, y, broadcast_helper);
  auto zero = XlaHelpers::Zero(b, dtype);
  auto trunc_mod = xla::Rem(x, y);
  auto trunc_mod_not_zero = xla::Ne(trunc_mod, zero);
  auto do_plus = xla::And(xla::Ne(xla::Lt(trunc_mod, zero), xla::Lt(y, zero)),
                          trunc_mod_not_zero);
  return xla::Select(do_plus, xla::Add(trunc_mod, y), trunc_mod);
}
MACHINA_MACHINA_XLA_MAKE_BINARY(FloorMod,
                FloorModImpl(b, input_type(0), lhs, rhs, broadcast_helper));

MACHINA_MACHINA_XLA_MAKE_BINARY(BitwiseAnd, xla::And(lhs, rhs, extend_dimensions));
MACHINA_MACHINA_XLA_MAKE_BINARY(BitwiseOr, xla::Or(lhs, rhs, extend_dimensions));
MACHINA_MACHINA_XLA_MAKE_BINARY(BitwiseXor, xla::Xor(lhs, rhs, extend_dimensions));

MACHINA_MACHINA_XLA_MAKE_BINARY(LeftShift, xla::ShiftLeft(lhs, rhs, extend_dimensions));
MACHINA_MACHINA_XLA_MAKE_BINARY(RightShift,
                (DataTypeIsUnsigned(ctx->input_type(0))
                     ? xla::ShiftRightLogical(lhs, rhs, extend_dimensions)
                     : xla::ShiftRightArithmetic(lhs, rhs, extend_dimensions)));

MACHINA_MACHINA_XLA_MAKE_BINARY(LogicalAnd, xla::And(lhs, rhs, extend_dimensions));
MACHINA_MACHINA_XLA_MAKE_BINARY(LogicalOr, xla::Or(lhs, rhs, extend_dimensions));
MACHINA_MACHINA_XLA_MAKE_BINARY(Mod, xla::Rem(lhs, rhs, extend_dimensions));
MACHINA_MACHINA_XLA_MAKE_BINARY(Maximum, xla::Max(lhs, rhs, extend_dimensions));
MACHINA_MACHINA_XLA_MAKE_BINARY(Minimum, xla::Min(lhs, rhs, extend_dimensions));
MACHINA_MACHINA_XLA_MAKE_BINARY(RealDiv, xla::Div(lhs, rhs, extend_dimensions));
MACHINA_MACHINA_XLA_MAKE_BINARY(ReciprocalGrad, xla::Neg(xla::Mul(rhs, xla::Mul(lhs, lhs))));
MACHINA_MACHINA_XLA_MAKE_BINARY(
    RsqrtGrad,
    xla::Mul((lhs * lhs) * lhs,
             xla::Div(rhs, XlaHelpers::IntegerLiteral(b, input_type(0), -2)),
             extend_dimensions));
MACHINA_MACHINA_XLA_MAKE_BINARY(
    SqrtGrad,
    xla::Div(xla::Mul(rhs, XlaHelpers::FloatLiteral(b, input_type(0), 0.5)),
             lhs, extend_dimensions));

// Implementation of TruncateDiv.
//
// For floating-point values, returns trunc(x / y).  For integers, simply
// returns x / y.
static xla::XlaOp TruncateDivImpl(xla::XlaBuilder* b, DataType dtype,
                                  xla::XlaOp x, xla::XlaOp y,
                                  const BCast& broadcast_helper) {
  std::tie(x, y) = XlaBinaryOp::Broadcast(x, y, broadcast_helper);
  if (!DataTypeIsFloating(dtype)) {
    return xla::Div(x, y);
  }
  auto zero = XlaHelpers::Zero(b, dtype);
  auto x_div_y = xla::Div(x, y);
  auto round_up = xla::Lt(x_div_y, zero);
  return xla::Select(round_up, xla::Ceil(x_div_y), xla::Floor(x_div_y));
}
MACHINA_MACHINA_XLA_MAKE_BINARY(TruncateDiv,
                TruncateDivImpl(b, input_type(0), lhs, rhs, broadcast_helper));
MACHINA_MACHINA_XLA_MAKE_BINARY(TruncateMod, xla::Rem(lhs, rhs, extend_dimensions));

// Comparison ops
MACHINA_MACHINA_XLA_MAKE_BINARY(Equal, xla::Eq(lhs, rhs, extend_dimensions));
MACHINA_MACHINA_XLA_MAKE_BINARY(NotEqual, xla::Ne(lhs, rhs, extend_dimensions));
MACHINA_MACHINA_XLA_MAKE_BINARY(Greater, xla::Gt(lhs, rhs, extend_dimensions));
MACHINA_MACHINA_XLA_MAKE_BINARY(GreaterEqual, xla::Ge(lhs, rhs, extend_dimensions));
MACHINA_MACHINA_XLA_MAKE_BINARY(Less, xla::Lt(lhs, rhs, extend_dimensions));
MACHINA_MACHINA_XLA_MAKE_BINARY(LessEqual, xla::Le(lhs, rhs, extend_dimensions));

// Non-linear ops
MACHINA_MACHINA_XLA_MAKE_BINARY(SigmoidGrad,
                xla::Mul(xla::Mul(rhs, lhs),
                         xla::Sub(XlaHelpers::One(b, input_type(0)), lhs)));

MACHINA_MACHINA_XLA_MAKE_BINARY(SoftplusGrad, xla::Mul(lhs, xla::Logistic(rhs)));

// softsigngrad(gradients, features) = gradients / (1 + abs(features)) ** 2
MACHINA_MACHINA_XLA_MAKE_BINARY(SoftsignGrad,
                xla::Div(lhs,
                         xla::Square(xla::Add(XlaHelpers::One(b, input_type(0)),
                                              xla::Abs(rhs)))));

MACHINA_MACHINA_XLA_MAKE_BINARY(TanhGrad,
                xla::Mul(rhs, xla::Sub(XlaHelpers::One(b, input_type(0)),
                                       xla::Mul(lhs, lhs))));

MACHINA_MACHINA_XLA_MAKE_BINARY(Pow, xla::Pow(lhs, rhs, extend_dimensions));

xla::XlaOp SquaredDifferenceImpl(
    DataType dtype, xla::XlaOp x, xla::XlaOp y,
    const std::vector<int64_t>& extend_dimensions) {
  auto difference = xla::Sub(x, y, extend_dimensions);
  if (DataTypeIsComplex(dtype)) {
    return xla::Conj(difference) * difference;
  } else {
    return xla::Square(difference);
  }
}
MACHINA_MACHINA_XLA_MAKE_BINARY(SquaredDifference,
                SquaredDifferenceImpl(input_type(0), lhs, rhs,
                                      extend_dimensions));

xla::XlaOp IgammaImpl(xla::XlaOp x, xla::XlaOp y,
                      const BCast& broadcast_helper) {
  std::tie(x, y) = XlaBinaryOp::Broadcast(x, y, broadcast_helper);
  return xla::Igamma(x, y);
}

MACHINA_MACHINA_XLA_MAKE_BINARY(Igamma, IgammaImpl(lhs, rhs, broadcast_helper));

xla::XlaOp IgammaGradAImpl(xla::XlaOp x, xla::XlaOp y,
                           const BCast& broadcast_helper) {
  std::tie(x, y) = XlaBinaryOp::Broadcast(x, y, broadcast_helper);
  return xla::IgammaGradA(x, y);
}

MACHINA_MACHINA_XLA_MAKE_BINARY(IgammaGradA, IgammaGradAImpl(lhs, rhs, broadcast_helper));

xla::XlaOp RandomGammaGradImpl(xla::XlaOp x, xla::XlaOp y,
                               const BCast& broadcast_helper) {
  std::tie(x, y) = XlaBinaryOp::Broadcast(x, y, broadcast_helper);
  return xla::RandomGammaGrad(x, y);
}

MACHINA_MACHINA_XLA_MAKE_BINARY(RandomGammaGrad,
                RandomGammaGradImpl(lhs, rhs, broadcast_helper));

xla::XlaOp IgammacImpl(xla::XlaOp x, xla::XlaOp y,
                       const BCast& broadcast_helper) {
  std::tie(x, y) = XlaBinaryOp::Broadcast(x, y, broadcast_helper);
  return xla::Igammac(x, y);
}

MACHINA_MACHINA_XLA_MAKE_BINARY(Igammac, IgammacImpl(lhs, rhs, broadcast_helper));

xla::XlaOp PolygammaImpl(xla::XlaOp n, xla::XlaOp x,
                         const BCast& broadcast_helper) {
  std::tie(n, x) = XlaBinaryOp::Broadcast(n, x, broadcast_helper);
  return xla::Polygamma(n, x);
}

MACHINA_MACHINA_XLA_MAKE_BINARY(Polygamma, PolygammaImpl(lhs, rhs, broadcast_helper));

xla::XlaOp ZetaImpl(xla::XlaOp x, xla::XlaOp q, const BCast& broadcast_helper) {
  std::tie(x, q) = XlaBinaryOp::Broadcast(x, q, broadcast_helper);
  return xla::Zeta(x, q);
}

MACHINA_MACHINA_XLA_MAKE_BINARY(Zeta, ZetaImpl(lhs, rhs, broadcast_helper));

#undef MACHINA_MACHINA_XLA_MAKE_BINARY

class ApproximateEqualOp : public XlaOpKernel {
 public:
  explicit ApproximateEqualOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tolerance", &tolerance_));
  }

  // Computes the max of the scalar input x and 0.
  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    auto abs = xla::Abs(xla::Sub(ctx->Input(0), ctx->Input(1)));
    auto abs_shape = b->GetShape(abs);
    OP_REQUIRES_OK(ctx, abs_shape.status());
    auto abs_type = abs_shape.value().element_type();
    auto result =
        xla::Lt(abs, xla::ConvertElementType(
                         xla::ConstantR0<float>(b, tolerance_), abs_type));
    ctx->SetOutput(0, result);
  }

 private:
  float tolerance_;
};
REGISTER_MACHINA_MACHINA_XLA_OP(Name("ApproximateEqual"), ApproximateEqualOp);

}  // namespace
}  // namespace machina
