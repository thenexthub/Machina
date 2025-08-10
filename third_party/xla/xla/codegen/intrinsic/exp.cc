/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "machina/xla/codegen/intrinsic/exp.h"

#include <limits>

#include "absl/log/check.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/Analysis/CGSCCPassManager.h"
#include "toolchain/Analysis/LoopAnalysisManager.h"
#include "toolchain/Analysis/TargetLibraryInfo.h"
#include "toolchain/ExecutionEngine/ExecutionEngine.h"
#include "toolchain/ExecutionEngine/Orc/IRCompileLayer.h"
#include "toolchain/IR/Argument.h"
#include "toolchain/IR/Constants.h"
#include "toolchain/IR/DerivedTypes.h"
#include "toolchain/IR/IRBuilder.h"
#include "toolchain/IR/Instructions.h"
#include "toolchain/IR/Intrinsics.h"
#include "toolchain/IR/LegacyPassManager.h"
#include "toolchain/IR/Module.h"
#include "toolchain/IR/PassManager.h"
#include "toolchain/IR/Type.h"
#include "toolchain/IR/Value.h"
#include "toolchain/Passes/StandardInstrumentations.h"
#include "toolchain/Support/Casting.h"
#include "toolchain/Support/TypeSize.h"
#include "toolchain/Target/TargetMachine.h"
#include "toolchain/Transforms/IPO/GlobalDCE.h"
#include "toolchain/Transforms/Utils/BuildLibCalls.h"
#include "toolchain/Transforms/Utils/ModuleUtils.h"
#include "machina/xla/codegen/intrinsic/intrinsic.h"
#include "machina/xla/codegen/intrinsic/ldexp.h"

namespace xla::codegen::intrinsics {

// Creates an LLVM function that implements a vectorized exponential function
// (exp(x)). The implementation uses a polynomial approximation method based on
// https://gitlab.com/libeigen/eigen/-/blob/21e89b930c6af56dbdaeea2a91d8b9d6fd2c208a/Eigen/src/Core/arch/Default/GenericPacketMathFunctions.h#L645
absl::StatusOr<toolchain::Function*> Exp::CreateDefinition(toolchain::Module* module,
                                                      Type type) {
  toolchain::Type* input_type = Type::TypeToIrType(type, module->getContext());
  CHECK(input_type != nullptr);
  CHECK(input_type->isFloatingPointTy() || input_type->isVectorTy())
      << "Vector type must be a floating point or vector of floating point.";
  CHECK(input_type->getScalarType()->isDoubleTy())
      << "Only F64 (double) is supported for xla.exp.";
  toolchain::LLVMContext& context = module->getContext();
  toolchain::IRBuilder<> builder(context);

  int num_elements = 1;
  if (toolchain::VectorType* vec_ty = toolchain::dyn_cast<toolchain::VectorType>(input_type)) {
    num_elements = vec_ty->getElementCount().getKnownMinValue();
  }

  toolchain::FunctionType* function_type =
      toolchain::FunctionType::get(input_type, {input_type}, false);
  toolchain::Function* func = toolchain::dyn_cast<toolchain::Function>(
      module->getOrInsertFunction(Exp::Name(type), function_type).getCallee());

  toolchain::Argument* input_x_arg = func->getArg(0);
  input_x_arg->setName("input_x");

  toolchain::BasicBlock* entry_bb = toolchain::BasicBlock::Create(context, "entry", func);
  builder.SetInsertPoint(entry_bb);

  const auto v_const = [&](double val) {
    return toolchain::ConstantFP::get(input_type, val);
  };

  toolchain::Constant* kVecZero = v_const(0.0);
  toolchain::Constant* kVecOne = v_const(1.0);
  toolchain::Constant* kVecTwo = v_const(2.0);
  toolchain::Constant* kVecHalf = v_const(0.5);

  // Upper and lower bounds for the input argument to avoid overflow/underflow
  // during intermediate calculations.
  toolchain::Constant* kVecExpHi =
      v_const(709.782712893384);  // exp(709.78) ~ DBL_MAX
  toolchain::Constant* kVecExpLo =
      v_const(-708.3964185322641);  // exp(-708.39) ~ DBL_MIN normal

  // Constants for the rational polynomial approximation of exp(x)
  // P(x)/Q(x) for x in [-ln(2)/2, ln(2)/2]
  // These are typically high-precision (double) constants.
  toolchain::Constant* kVecLog2ef = v_const(1.4426950408889634073599);  // log2(e)
  toolchain::Constant* kVecPolyP0 = v_const(1.26177193074810590878e-4);
  toolchain::Constant* kVecPolyP1 = v_const(3.02994407707441961300e-2);
  toolchain::Constant* kVecPolyP2 = v_const(9.99999999999999999910e-1);
  toolchain::Constant* kVecPolyQ0 = v_const(3.00198505138664455042e-6);
  toolchain::Constant* kVecPolyQ1 = v_const(2.52448340349684104192e-3);
  toolchain::Constant* kVecPolyQ2 = v_const(2.27265548208155028766e-1);
  toolchain::Constant* kVecPolyQ3 = v_const(2.00000000000000000009e0);

  // Constants for argument reduction x = n*ln(2) + g
  // C1 and C2 are parts of ln(2) for higher precision: C1 + C2 = ln(2)
  toolchain::Constant* kVecLog2C1 = v_const(0.693145751953125);          // ln(2) C1
  toolchain::Constant* kVecLog2C2 = v_const(1.42860682030941723212e-6);  // ln(2) C2

  toolchain::Value* x_input_alias = input_x_arg;  // Alias for clarity

  // Create a mask for inputs that are below the lower operational limit.
  // These might directly map to zero later.
  toolchain::Value* is_below_exp_lo_mask =
      builder.CreateFCmpOLT(x_input_alias, kVecExpLo, "is_below_exp_lo_mask");

  toolchain::Value* x_clamped = x_input_alias;
  // Clamp the working copy of x to the operational range [exp_lo, exp_hi].
  toolchain::Function* minimum_fn = toolchain::Intrinsic::getOrInsertDeclaration(
      module, toolchain::Intrinsic::minimum, {input_type});
  x_clamped = builder.CreateCall(minimum_fn, {x_clamped, kVecExpHi});
  toolchain::Function* maximum_fn = toolchain::Intrinsic::getOrInsertDeclaration(
      module, toolchain::Intrinsic::maximum, {input_type});
  x_clamped = builder.CreateCall(maximum_fn, {x_clamped, kVecExpLo});

  // Argument reduction: express exp(x) as 2^n * exp(g)
  // Calculate n = round_to_nearest_integer(x / ln(2)) = floor(x * log2(e) +
  // 0.5)
  toolchain::Value* n = builder.CreateFMul(x_clamped, kVecLog2ef, "x_mul_log2ef");
  n = builder.CreateFAdd(n, kVecHalf, "add_half_for_round");

  toolchain::Function* floor_fn = toolchain::Intrinsic::getOrInsertDeclaration(
      module, toolchain::Intrinsic::floor, {input_type});
  n = builder.CreateCall(floor_fn, {n}, "n_float_val");

  // Calculate g = x - n * ln(2).
  // To maintain precision, ln(2) is split into C1 and C2.
  // g = x_clamped - (n * kVecLog2C1 + n * kVecLog2C2)
  toolchain::Value* n_mul_c1 = builder.CreateFMul(n, kVecLog2C1, "n_mul_log2c1");
  toolchain::Value* n_mul_c2 = builder.CreateFMul(n, kVecLog2C2, "n_mul_log2c2");

  toolchain::Value* g_vec =
      builder.CreateFSub(x_clamped, n_mul_c1, "g_intermediate_sub_c1");
  g_vec = builder.CreateFSub(g_vec, n_mul_c2, "g_reduced_arg");
  // g_vec now holds g, where -ln(2)/2 <= g <= ln(2)/2.

  toolchain::Value* g_squared_vec = builder.CreateFMul(g_vec, g_vec, "g_squared");

  // Evaluate the numerator polynomial P(g^2) for the rational approximation
  // using Horner's method, then multiply by g. P(g) = ((P0 * g^2 + P1) * g^2 +
  // P2) * g.
  toolchain::Value* p_num_poly = kVecPolyP0;
  p_num_poly = builder.CreateFMul(p_num_poly, g_squared_vec, "p_poly_term0");
  p_num_poly = builder.CreateFAdd(p_num_poly, kVecPolyP1, "p_poly_term1");
  p_num_poly = builder.CreateFMul(p_num_poly, g_squared_vec, "p_poly_term2");
  p_num_poly = builder.CreateFAdd(p_num_poly, kVecPolyP2, "p_poly_term3");
  p_num_poly = builder.CreateFMul(p_num_poly, g_vec, "p_poly_final_mul_g");

  // Evaluate the denominator polynomial Q(g^2) using Horner's method.
  // Q(g^2) = ((Q0 * g^2 + Q1) * g^2 + Q2) * g^2 + Q3.
  toolchain::Value* q_den_poly = kVecPolyQ0;
  q_den_poly = builder.CreateFMul(q_den_poly, g_squared_vec, "q_poly_term0");
  q_den_poly = builder.CreateFAdd(q_den_poly, kVecPolyQ1, "q_poly_term1");
  q_den_poly = builder.CreateFMul(q_den_poly, g_squared_vec, "q_poly_term2");
  q_den_poly = builder.CreateFAdd(q_den_poly, kVecPolyQ2, "q_poly_term3");
  q_den_poly = builder.CreateFMul(q_den_poly, g_squared_vec, "q_poly_term4");
  q_den_poly = builder.CreateFAdd(q_den_poly, kVecPolyQ3, "q_poly_final");

  // Rational approximation for exp(g) part: exp(g) ~ 1 + 2*g * P(g^2) / (Q(g^2)
  // - g*P(g^2)) The formula used by Cephes (from which constants are likely
  // derived) is: exp(g) ~ 1 + 2 * P(g) / (Q(g) - P(g))  (where P and Q are
  // polynomials in g, not g^2 directly) Or, based on the variable names in the
  // original Eigen code's structure: exp_g_approx_term = P_num(g) / (Q_den(g) -
  // P_num(g)) exp_g_approx = 1.0 + 2.0 * exp_g_approx_term
  toolchain::Value* q_minus_p =
      builder.CreateFSub(q_den_poly, p_num_poly, "q_poly_sub_p_poly");
  toolchain::Value* exp_g_term =
      builder.CreateFDiv(p_num_poly, q_minus_p, "exp_g_rational_term");

  toolchain::Value* exp_g_approx =
      builder.CreateFMul(exp_g_term, kVecTwo, "exp_g_term_mul_2");
  exp_g_approx =
      builder.CreateFAdd(exp_g_approx, kVecOne, "exp_g_approx_final");
  // exp_g_approx now holds the approximation for exp(g).

  // Convert n (stored as float vector) to an integer vector for ldexp.
  Type ldexp_int_type = Type(xla::S32, type.vector_width());
  toolchain::Function* ldexp_fn =
      Ldexp::CreateDefinition(module, type, ldexp_int_type).value();
  // FPtoSI(nan) yields a poison value. We freeze the output to halt propagation
  // of UB and let the compiler know we will accept any arbitrary value of that
  // type here.
  // This works because ldexp(nan, n_int) = nan for any n_int.
  toolchain::Value* n_int = builder.CreateFreeze(builder.CreateFPToSI(
      n, ldexp_fn->getArg(1)->getType(), "n_float_to_int"));

  // Reconstruct exp(x) = exp(g) * 2^n using ldexp(exp_g_approx, n_int_vec).
  toolchain::Value* calculated_exp_val = builder.CreateCall(
      ldexp_fn, {exp_g_approx, n_int}, "calculated_exp_val_ldexp");

  // --- Final Selection for Out-of-Range Inputs ---
  // The main calculation is only valid for inputs within the [lo, hi] range.
  // For inputs outside this range, we explicitly select the correct result
  // (0.0 for underflow, +inf for overflow).

  toolchain::Constant* kVecInf = v_const(std::numeric_limits<double>::infinity());
  toolchain::Value* is_above_exp_hi_mask =
      builder.CreateFCmpOGT(input_x_arg, kVecExpHi, "is_above_exp_hi_mask");
  toolchain::Value* result_with_underflow =
      builder.CreateSelect(is_below_exp_lo_mask, kVecZero, calculated_exp_val,
                           "result_with_underflow");
  toolchain::Value* final_result =
      builder.CreateSelect(is_above_exp_hi_mask, kVecInf, result_with_underflow,
                           "final_result_with_overflow");

  builder.CreateRet(final_result);
  return func;
}

}  // namespace xla::codegen::intrinsics
