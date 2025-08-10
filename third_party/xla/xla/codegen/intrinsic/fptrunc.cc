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

#include "machina/xla/codegen/intrinsic/fptrunc.h"

#include "absl/log/check.h"
#include "toolchain/ADT/FloatingPointMode.h"
#include "toolchain/IR/Argument.h"
#include "toolchain/IR/BasicBlock.h"
#include "toolchain/IR/Constants.h"
#include "toolchain/IR/DerivedTypes.h"
#include "toolchain/IR/Function.h"
#include "toolchain/IR/IRBuilder.h"
#include "toolchain/IR/Module.h"
#include "toolchain/IR/Type.h"
#include "toolchain/Support/Casting.h"
#include "machina/xla/codegen/intrinsic/intrinsic.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/xla/util.h"
#include "machina/xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {
namespace {
toolchain::Function* CreateFunction(toolchain::Module* module, Type from, Type to) {
  DCHECK_OK(Type::VerifySameWidth(from, to));
  toolchain::LLVMContext& context = module->getContext();
  toolchain::FunctionType* function_type = toolchain::FunctionType::get(
      to.to_ir_type(context), {from.to_ir_type(context)},
      /*isVarArg=*/false);
  toolchain::Function* func = toolchain::dyn_cast<toolchain::Function>(
      module->getOrInsertFunction(FpTrunc::Name(from, to), function_type)
          .getCallee());
  func->getArg(0)->setName("arg");
  return func;
}
}  // namespace

// Truncates an f32 value (scalar or vector) to bf16 with correct rounding.
static toolchain::Function* TruncateF32ToBf16(toolchain::Module* module, Type from,
                                         Type to) {
  toolchain::LLVMContext& context = module->getContext();
  toolchain::IRBuilder<> builder(context);
  toolchain::Function* func = CreateFunction(module, from, to);

  // Wraps a scalar type into a vector type if we are building a vector
  // intrinsic declaration.
  auto vec = [&](toolchain::Type* scalar_type) -> toolchain::Type* {
    if (from.vector_width().has_value()) {
      return toolchain::VectorType::get(scalar_type, *from.vector_width(), false);
    }
    return scalar_type;
  };

  toolchain::Type* i16_type = vec(builder.getInt16Ty());
  toolchain::Type* i32_type = vec(builder.getInt32Ty());
  toolchain::Type* bf16_type = vec(builder.getBFloatTy());

  toolchain::Argument* arg = func->getArg(0);
  toolchain::BasicBlock* entry_bb = toolchain::BasicBlock::Create(context, "entry", func);
  builder.SetInsertPoint(entry_bb);

  auto* i32 = builder.CreateBitCast(arg, i32_type);

  // Rounding bias for non-nan values.
  auto* lsb = builder.CreateAnd(builder.CreateLShr(i32, 16),
                                toolchain::ConstantInt::get(i32_type, 1));
  auto* rounding_bias =
      builder.CreateAdd(toolchain::ConstantInt::get(i32_type, 0x7fff), lsb);

  // For NaNs, we set all of them to quiet NaNs by masking the mantissa
  // so that only the MSB is 1, then simply truncate the original value
  // to retain the sign.
  auto* is_nan = builder.createIsFPClass(arg, toolchain::FPClassTest::fcNan);
  auto* nan_mask = toolchain::ConstantInt::get(i32_type, 0xFFC00000);
  auto* msb = toolchain::ConstantInt::get(i32_type, 0x00400000);
  auto* quiet_nan = builder.CreateOr(builder.CreateAnd(i32, nan_mask), msb);
  auto* i16 = builder.CreateTrunc(
      builder.CreateLShr(
          builder.CreateSelect(is_nan, quiet_nan,
                               builder.CreateAdd(i32, rounding_bias)),
          16),
      i16_type);

  toolchain::Value* result = builder.CreateBitCast(i16, bf16_type);
  builder.CreateRet(result);

  return func;
}

static toolchain::Function* ExtendF8e5m2ToF16(toolchain::Module* module, Type from,
                                         Type to) {
  toolchain::LLVMContext& context = module->getContext();
  toolchain::IRBuilder<> builder(context);
  toolchain::Function* func = CreateFunction(module, from, to);
  toolchain::BasicBlock* entry_bb = toolchain::BasicBlock::Create(context, "entry", func);
  builder.SetInsertPoint(entry_bb);

  toolchain::Value* as_int16 = builder.CreateZExt(
      func->getArg(0), Type(S16, from.vector_width()).to_ir_type(context));
  toolchain::Value* shifted = builder.CreateShl(as_int16, 8);
  builder.CreateRet(builder.CreateBitCast(shifted, to.to_ir_type(context)));
  return func;
}

absl::StatusOr<toolchain::Function*> FpTrunc::CreateDefinition(toolchain::Module* module,
                                                          Type from, Type to) {
  TF_RETURN_IF_ERROR(Type::VerifySameWidth(from, to));

  if (from.element_type() == F32 && to.element_type() == BF16) {
    return TruncateF32ToBf16(module, from, to);
  }
  if (from.element_type() == F8E5M2 && to.element_type() == F16) {
    return ExtendF8e5m2ToF16(module, from, to);
  }

  return Internal("Unsupported fptrunc conversion: from=%s to=%s", from.name(),
                  to.name());
}

}  // namespace xla::codegen::intrinsics
