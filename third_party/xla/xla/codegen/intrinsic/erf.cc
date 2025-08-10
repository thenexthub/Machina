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

#include "machina/xla/codegen/intrinsic/erf.h"

#include "absl/log/check.h"
#include "toolchain/IR/Argument.h"
#include "toolchain/IR/BasicBlock.h"
#include "toolchain/IR/Constants.h"
#include "toolchain/IR/DerivedTypes.h"
#include "toolchain/IR/Function.h"
#include "toolchain/IR/IRBuilder.h"
#include "toolchain/IR/Intrinsics.h"
#include "toolchain/IR/Module.h"
#include "toolchain/IR/Type.h"
#include "toolchain/Support/Casting.h"
#include "machina/xla/codegen/intrinsic/intrinsic.h"
#include "machina/xla/service/llvm_ir/llvm_util.h"

namespace xla::codegen::intrinsics {

// Emits an approximation of erf. The implementation uses the same rational
// interpolant as implemented in Eigen3.
static toolchain::Value* EmitErfF32(toolchain::IRBuilderBase* b, toolchain::Value* x) {
  auto type = x->getType();
  constexpr float kErfInvOneMinusHalfULP = 3.832506856900711f;
  auto call_fabs = [b](toolchain::Value* operand_value) {
    return llvm_ir::EmitCallToIntrinsic(toolchain::Intrinsic::fabs, {operand_value},
                                        {operand_value->getType()}, b);
  };
  auto fcmp_le = [b](toolchain::Value* lhs_value, toolchain::Value* rhs_value) {
    return b->CreateFCmpOLE(lhs_value, rhs_value);
  };
  toolchain::Value* const clamp = fcmp_le(
      toolchain::ConstantFP::get(type, kErfInvOneMinusHalfULP), call_fabs(x));
  // The monomial coefficients of the numerator polynomial (odd).
  toolchain::Value* const alpha_1 = toolchain::ConstantFP::get(type, 1.128379143519084f);
  toolchain::Value* const alpha_3 =
      toolchain::ConstantFP::get(type, 0.18520832239976145f);
  toolchain::Value* const alpha_5 =
      toolchain::ConstantFP::get(type, 0.050955695062380861f);
  toolchain::Value* const alpha_7 =
      toolchain::ConstantFP::get(type, 0.0034082910107109506f);
  toolchain::Value* const alpha_9 =
      toolchain::ConstantFP::get(type, 0.00022905065861350646f);

  // The monomial coefficients of the denominator polynomial (even).
  toolchain::Value* const beta_0 = toolchain::ConstantFP::get(type, 1.0f);
  toolchain::Value* const beta_2 = toolchain::ConstantFP::get(type, 0.49746925110067538f);
  toolchain::Value* const beta_4 = toolchain::ConstantFP::get(type, 0.11098505178285362f);
  toolchain::Value* const beta_6 =
      toolchain::ConstantFP::get(type, 0.014070470171167667f);
  toolchain::Value* const beta_8 =
      toolchain::ConstantFP::get(type, 0.0010179625278914885f);
  toolchain::Value* const beta_10 =
      toolchain::ConstantFP::get(type, 0.000023547966471313185f);
  toolchain::Value* const beta_12 =
      toolchain::ConstantFP::get(type, -1.1791602954361697e-7f);

  // Since the polynomials are odd/even, we need x^2.
  toolchain::Value* const x2 = b->CreateFMul(x, x);

  // Evaluate the numerator polynomial p.
  auto call_fma = [b](toolchain::Value* multiplier, toolchain::Value* multiplicand,
                      toolchain::Value* addend) {
    return llvm_ir::EmitCallToIntrinsic(toolchain::Intrinsic::fma,
                                        {multiplier, multiplicand, addend},
                                        {multiplier->getType()}, b);
  };
  toolchain::Value* p = call_fma(x2, alpha_9, alpha_7);
  p = call_fma(x2, p, alpha_5);
  p = call_fma(x2, p, alpha_3);
  p = call_fma(x2, p, alpha_1);
  p = b->CreateFMul(x, p);

  // Evaluate the denominator polynomial p.
  toolchain::Value* q = call_fma(x2, beta_12, beta_10);
  q = call_fma(x2, q, beta_8);
  q = call_fma(x2, q, beta_6);
  q = call_fma(x2, q, beta_4);
  q = call_fma(x2, q, beta_2);
  q = call_fma(x2, q, beta_0);

  // Divide the numerator by the denominator.
  auto call_copysign = [b](toolchain::Value* mag, toolchain::Value* sign) {
    return llvm_ir::EmitCallToIntrinsic(toolchain::Intrinsic::copysign, {mag, sign},
                                        {mag->getType()}, b);
  };
  auto* result =
      b->CreateSelect(clamp, call_copysign(toolchain::ConstantFP::get(type, 1.0), x),
                      b->CreateFDiv(p, q));
  return result;
}

absl::StatusOr<toolchain::Function*> Erf::CreateDefinition(
    toolchain::Module* module, const Type intrinsic_type) {
  toolchain::Type* type = Type::TypeToIrType(intrinsic_type, module->getContext());
  CHECK(type != nullptr);
  CHECK(type->isFloatTy() ||
        (type->isVectorTy() && type->getScalarType()->isFloatTy()))
      << "Type must be a f32 or vector of f32.";

  toolchain::LLVMContext& context = module->getContext();
  toolchain::IRBuilder<> builder(context);

  int num_elements = 1;
  if (toolchain::VectorType* vec_ty = toolchain::dyn_cast<toolchain::VectorType>(type)) {
    num_elements = vec_ty->getElementCount().getKnownMinValue();
  }

  toolchain::FunctionType* function_type =
      toolchain::FunctionType::get(type, {type}, false);
  toolchain::Function* func = toolchain::dyn_cast<toolchain::Function>(
      module->getOrInsertFunction(Name(intrinsic_type), function_type)
          .getCallee());

  toolchain::Argument* input_value = func->getArg(0);
  input_value->setName("input_value");

  toolchain::BasicBlock* entry_bb = toolchain::BasicBlock::Create(context, "entry", func);
  builder.SetInsertPoint(entry_bb);

  builder.CreateRet(EmitErfF32(&builder, input_value));

  return func;
}

}  // namespace xla::codegen::intrinsics
