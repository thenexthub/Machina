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

#include "machina/xla/codegen/intrinsic/ldexp.h"

#include <cstdint>

#include "absl/log/check.h"
#include "toolchain/IR/Attributes.h"
#include "toolchain/IR/BasicBlock.h"
#include "toolchain/IR/Constants.h"
#include "toolchain/IR/DerivedTypes.h"
#include "toolchain/IR/Function.h"
#include "toolchain/IR/IRBuilder.h"
#include "toolchain/IR/Instructions.h"
#include "toolchain/IR/Intrinsics.h"
#include "toolchain/IR/Module.h"
#include "toolchain/IR/Type.h"
#include "toolchain/IR/Value.h"
#include "toolchain/Support/Casting.h"
#include "toolchain/Support/TypeSize.h"
#include "machina/xla/codegen/intrinsic/intrinsic.h"
#include "machina/xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {

namespace {
toolchain::Value* IntMax(toolchain::IRBuilderBase& builder, toolchain::Value* v1,
                    toolchain::Value* v2) {
  toolchain::Value* cmp = builder.CreateICmpSGT(v1, v2);
  return builder.CreateSelect(cmp, v1, v2);
}

toolchain::Value* IntMin(toolchain::IRBuilderBase& builder, toolchain::Value* v1,
                    toolchain::Value* v2) {
  toolchain::Value* cmp = builder.CreateICmpSLT(v1, v2);
  return builder.CreateSelect(cmp, v1, v2);
}
}  // namespace

absl::StatusOr<toolchain::Function*> Ldexp::CreateDefinition(toolchain::Module* module,
                                                        Type base,
                                                        Type exp_type) {
  // This implementation closely follows Eigen's ldexp implementation:
  // https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Core/arch/Default/GenericPacketMathFunctions.h#L226
  // One key difference being that the 2nd exponent argument is an integer or
  // vector of integers, not doubles.

  toolchain::Type* vector_type = Type::TypeToIrType(base, module->getContext());
  toolchain::Type* input_int_type =
      Type::TypeToIrType(exp_type, module->getContext());

  CHECK(vector_type != nullptr);
  CHECK(vector_type->isFloatingPointTy() || vector_type->isVectorTy())
      << "Vector type must be a floating point or vector of floating point.";
  CHECK(vector_type->getScalarType()->isDoubleTy())
      << "Only F64 (double) is supported for ldexp.";

  int num_elements = 1;
  toolchain::Type* i64_type = toolchain::Type::getInt64Ty(module->getContext());

  if (toolchain::VectorType* vec_ty =
          toolchain::dyn_cast<toolchain::VectorType>(vector_type)) {
    num_elements = vec_ty->getElementCount().getKnownMinValue();
    i64_type = toolchain::VectorType::get(i64_type, num_elements, false);
  }

  toolchain::FunctionType* ldexp_func_type = toolchain::FunctionType::get(
      vector_type, {vector_type, input_int_type}, false);
  toolchain::Function* ldexp_func =
      toolchain::Function::Create(ldexp_func_type, toolchain::Function::InternalLinkage,
                             Ldexp::Name(base, exp_type), module);
  ldexp_func->addFnAttr(toolchain::Attribute::AlwaysInline);
  toolchain::Argument* a = ldexp_func->getArg(0);
  a->setName("a");
  toolchain::Argument* exponent = ldexp_func->getArg(1);
  exponent->setName("exponent");

  // 2. Create a basic block
  toolchain::BasicBlock* entry_block =
      toolchain::BasicBlock::Create(module->getContext(), "entry", ldexp_func);
  toolchain::IRBuilder<> builder = toolchain::IRBuilder<>(entry_block);

  auto int_vec = [=](int64_t val) {
    return toolchain::ConstantInt::get(i64_type, val, true);
  };

  // Constants for double (F64) based on IEEE 754 standard.
  static constexpr int kMantissaBits = 52;  // Excludes implicit leading '1'.
  static constexpr int kExponentBits = 11;  // And one left for sign.

  // Exponent bias for IEEE 754 double = 1023.
  toolchain::Value* bias_val = int_vec((1LL << (kExponentBits - 1)) - 1);

  toolchain::Value* max_exponent = toolchain::ConstantInt::get(i64_type, 2099);

  // Clamp the exponent: e = min(max(exponent, -max_exponent), max_exponent).
  toolchain::Value* neg_max_exponent = builder.CreateNeg(max_exponent);
  toolchain::Value* sext_exponent = builder.CreateSExt(exponent, i64_type);
  toolchain::Value* clamped_exponent =
      IntMax(builder, sext_exponent, neg_max_exponent);
  clamped_exponent = IntMin(builder, clamped_exponent, max_exponent);

  toolchain::Value* two_i64_for_shift = int_vec(2);
  // floor(e/4):
  toolchain::Value* b = builder.CreateAShr(clamped_exponent, two_i64_for_shift, "b");

  // Calculate 2^b (first factor 'c') using bit manipulation:
  //    a. Add `b` to the exponent `bias` (integer addition).
  //    b. Perform a logical shift left to position the
  //       new exponent value correctly within the 64-bit integer representing
  //       the floating-point number.
  //    c. Bitcast the resulting integer bit pattern to a double.
  toolchain::Value* b_plus_bias = builder.CreateAdd(b, bias_val);
  toolchain::Value* mantissa_shift = int_vec(kMantissaBits);
  toolchain::Value* c_bits = builder.CreateShl(b_plus_bias, mantissa_shift);
  toolchain::Value* c = builder.CreateBitCast(c_bits, vector_type);

  // Calculate `out = a * 2^(3b)` which is `a * c * c * c`.
  toolchain::Value* out = builder.CreateFMul(a, c);
  out = builder.CreateFMul(out, c);
  out = builder.CreateFMul(out, c);

  // Calculate the remaining exponent adjustment: `b = e - 3*b`.
  toolchain::Value* three_b = builder.CreateMul(int_vec(3), b);
  b = builder.CreateSub(clamped_exponent, three_b);

  // Calculate `2^(e-3b)` (the second scaling factor 'c').
  b_plus_bias = builder.CreateAdd(b, bias_val);
  c_bits = builder.CreateShl(b_plus_bias, mantissa_shift);
  c = builder.CreateBitCast(c_bits, vector_type);
  out = builder.CreateFMul(out, c);
  builder.CreateRet(out);

  return ldexp_func;
}

}  // namespace xla::codegen::intrinsics
