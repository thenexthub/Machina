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

#include "machina/xla/codegen/intrinsic/rsqrt.h"

#include <cstddef>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "toolchain/ADT/APInt.h"
#include "toolchain/IR/Argument.h"
#include "toolchain/IR/BasicBlock.h"
#include "toolchain/IR/Constants.h"
#include "toolchain/IR/DerivedTypes.h"
#include "toolchain/IR/Function.h"
#include "toolchain/IR/IRBuilder.h"
#include "toolchain/IR/Instructions.h"
#include "toolchain/IR/Intrinsics.h"
#include "toolchain/IR/IntrinsicsX86.h"
#include "toolchain/IR/LLVMContext.h"
#include "toolchain/IR/Module.h"
#include "toolchain/IR/Type.h"
#include "toolchain/IR/Value.h"
#include "toolchain/Support/Casting.h"
#include "toolchain/Support/TypeSize.h"
#include "toolchain/Target/TargetMachine.h"
#include "machina/xla/codegen/intrinsic/intrinsic.h"
#include "machina/xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {
namespace {
toolchain::Value* PMAdd(toolchain::IRBuilder<>& builder, toolchain::Value* x, toolchain::Value* y,
                   toolchain::Value* z) {
  return builder.CreateFAdd(builder.CreateFMul(x, y), z);
}
}  // namespace

static toolchain::Value* NewtonRaphsonRsqrtIteration(toolchain::IRBuilder<>& builder,
                                                toolchain::Value* x,
                                                toolchain::Value* guess,
                                                toolchain::Type* type, int steps) {
  // Based on https://libeigen.gitlab.io/docs/MathFunctionsImpl_8h_source.html
  toolchain::Value* minus_half = toolchain::ConstantFP::get(type, -0.5);
  toolchain::Value* minus_one = toolchain::ConstantFP::get(type, -1.0);
  toolchain::Value* inv_sqrt = guess;
  for (int step = 0; step < steps; ++step) {
    // Refine the guess using one Newton-Raphson step.
    // h_n = (x * inv_sqrt) * inv_sqrt - 1 (so that h_n is nearly 0).
    // inv_sqrt = inv_sqrt - 0.5 * inv_sqrt * h_n
    toolchain::Value* r2 = builder.CreateFMul(x, inv_sqrt);
    toolchain::Value* half_r = builder.CreateFMul(inv_sqrt, minus_half);
    toolchain::Value* h_n = PMAdd(builder, r2, inv_sqrt, minus_one);
    inv_sqrt = PMAdd(builder, half_r, h_n, inv_sqrt);
  }
  return inv_sqrt;
}

struct RsqrtIntrinsic {
  toolchain::Intrinsic::ID id;
  int mask_bits;                  // Some avx512 calls require masks.
  int needs_insert_element_size;  // Some avx512 calls require padding.

  static RsqrtIntrinsic ForF32(size_t num_elements) {
    switch (num_elements) {
      case 1:
        return {toolchain::Intrinsic::x86_sse_rsqrt_ss, 0, 4};
      case 4:
        return {toolchain::Intrinsic::x86_sse_rsqrt_ps, 0, 0};
      case 8:
        return {toolchain::Intrinsic::x86_avx_rsqrt_ps_256, 0, 0};
      case 16:
        return {toolchain::Intrinsic::x86_avx512_rsqrt14_ps_512, 16, 0};
      default:
        LOG(FATAL) << "Unsupported vector width for rsqrt: " << num_elements;
    }
  }

  static RsqrtIntrinsic ForF64(size_t num_elements) {
    // We assume AVX512 is available for F64.
    switch (num_elements) {
      case 1:
        // Assuming AVX512 is available.
        // We don't use x86_avx512_rsqrt14_sd because it also requires padding
        // into <2 x double> vectors and it takes an additional source vector
        // for the upper bits of the result.
        return {toolchain::Intrinsic::x86_avx512_rsqrt14_pd_128, 8, 2};
      case 2:
        return {toolchain::Intrinsic::x86_avx512_rsqrt14_pd_128, 8, 0};
      case 4:
        return {toolchain::Intrinsic::x86_avx512_rsqrt14_pd_256, 8, 0};
      case 8:
        return {toolchain::Intrinsic::x86_avx512_rsqrt14_pd_512, 8, 0};
      default:
        LOG(FATAL) << "Unsupported vector width for rsqrt: " << num_elements;
    }
  }

  toolchain::Value* CreateCall(toolchain::IRBuilder<>& builder, toolchain::Value* x) {
    toolchain::Module* module = builder.GetInsertBlock()->getModule();
    toolchain::Function* rsqrt_intrinsic =
        toolchain::Intrinsic::getOrInsertDeclaration(module, id);

    toolchain::Value* y_approx;
    std::vector<toolchain::Value*> args = {x};
    if (needs_insert_element_size > 0) {
      // Pad into a vector of size `needs_insert_element_size`.
      toolchain::Type* sse_vec_type = toolchain::VectorType::get(
          x->getType()->getScalarType(),
          toolchain::ElementCount::getFixed(needs_insert_element_size));
      toolchain::Value* vec_x = toolchain::UndefValue::get(sse_vec_type);
      vec_x = builder.CreateInsertElement(vec_x, x, builder.getInt32(0));
      args[0] = vec_x;
    }
    if (mask_bits > 0) {
      toolchain::Value* src = toolchain::ConstantFP::get(args[0]->getType(), 0.0);
      toolchain::Value* mask = toolchain::ConstantInt::get(
          builder.getContext(), toolchain::APInt(mask_bits, -1, true));
      args.push_back(src);
      args.push_back(mask);
    }
    y_approx = builder.CreateCall(rsqrt_intrinsic, args, "y_approx");
    if (needs_insert_element_size > 0) {
      // Extract the result from the padded vector.
      y_approx = builder.CreateExtractElement(y_approx, builder.getInt32(0),
                                              "y_approx");
    }
    return y_approx;
  }
};

absl::StatusOr<toolchain::Function*> Rsqrt::CreateDefinition(
    toolchain::Module* module, absl::string_view features, Type type) {
  CHECK(type.element_type() == F64 || type.element_type() == F32)
      << type.name();
  toolchain::Type* input_type = Type::TypeToIrType(type, module->getContext());
  CHECK(input_type != nullptr);

  toolchain::LLVMContext& context = module->getContext();
  toolchain::IRBuilder<> builder(context);

  int num_elements = 1;
  if (toolchain::VectorType* vec_ty = toolchain::dyn_cast<toolchain::VectorType>(input_type)) {
    num_elements = vec_ty->getElementCount().getKnownMinValue();
  }

  toolchain::FunctionType* function_type =
      toolchain::FunctionType::get(input_type, {input_type}, false);
  toolchain::Function* func = toolchain::dyn_cast<toolchain::Function>(
      module->getOrInsertFunction(Rsqrt::Name(type), function_type)
          .getCallee());

  toolchain::Argument* x = func->getArg(0);
  x->setName("x");
  toolchain::BasicBlock* entry_bb = toolchain::BasicBlock::Create(context, "entry", func);
  builder.SetInsertPoint(entry_bb);

  if ((type.element_type() == F64 &&
       !absl::StrContains(features, "+avx512f")) ||
      !absl::StrContains(features, "+avx")) {
    LOG_EVERY_N(INFO, 1000)
        << "Falling back to 1 / sqrt(x) for " << type.name();
    // We can't use the same approximation algorithm for F64 without AVX512 or
    // anything non-x86 and without avx.
    toolchain::Value* one = toolchain::ConstantFP::get(input_type, 1.0);
    toolchain::Value* sqrt_x =
        builder.CreateUnaryIntrinsic(toolchain::Intrinsic::sqrt, x);
    toolchain::Value* inv_sqrt_x = builder.CreateFDiv(one, sqrt_x, "inv_sqrt_x");
    builder.CreateRet(inv_sqrt_x);
    return func;
  }

  RsqrtIntrinsic rsqrt_intrinsic = input_type->getScalarType()->isFloatTy()
                                       ? RsqrtIntrinsic::ForF32(num_elements)
                                       : RsqrtIntrinsic::ForF64(num_elements);
  toolchain::Value* y_approx = rsqrt_intrinsic.CreateCall(builder, x);

  // Eigen only does 1 step for F32, but that only gives within 2 ULPs and we
  // are targeting 1.
  const size_t num_steps = 2;
  toolchain::Value* refined_result =
      NewtonRaphsonRsqrtIteration(builder, x, y_approx, input_type, num_steps);

  const toolchain::fltSemantics& semantics =
      input_type->getScalarType()->getFltSemantics();
  toolchain::APFloat flt_min_val = toolchain::APFloat::getSmallestNormalized(semantics);
  toolchain::Constant* flt_min = toolchain::ConstantFP::get(input_type, flt_min_val);

  toolchain::Constant* inf =
      toolchain::ConstantFP::get(input_type, toolchain::APFloat::getInf(semantics));

  toolchain::Value* lt_min_mask = builder.CreateFCmpOLT(x, flt_min, "lt_min_mask");
  toolchain::Value* inf_mask = builder.CreateFCmpOEQ(x, inf, "inf_mask");
  toolchain::Value* use_hw_approx_mask =
      builder.CreateOr(lt_min_mask, inf_mask, "use_hw_approx_mask");

  toolchain::Value* result = builder.CreateSelect(use_hw_approx_mask, y_approx,
                                             refined_result, "result");

  builder.CreateRet(result);
  return func;
}

}  // namespace xla::codegen::intrinsics
