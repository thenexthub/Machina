/* Copyright 2024 The OpenXLA Authors.

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

#include "machina/xla/service/cpu/elemental_ir_emitter.h"

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "toolchain/IR/Constants.h"
#include "toolchain/IR/DerivedTypes.h"
#include "toolchain/IR/Instructions.h"
#include "toolchain/IR/Intrinsics.h"
#include "toolchain/IR/Value.h"
#include "machina/xla/codegen/intrinsic/exp.h"
#include "machina/xla/codegen/intrinsic/intrinsic.h"
#include "machina/xla/codegen/intrinsic/rsqrt.h"
#include "machina/xla/codegen/intrinsic/tanh.h"
#include "machina/xla/hlo/ir/hlo_computation.h"
#include "machina/xla/service/cpu/elemental_math_emitter.h"
#include "machina/xla/service/llvm_ir/llvm_util.h"

namespace xla::cpu {
using ::xla::codegen::intrinsics::Type;

absl::StatusOr<toolchain::Value*> CpuElementalIrEmitter::EmitAtan2(
    PrimitiveType prim_type, toolchain::Value* lhs, toolchain::Value* rhs,
    absl::string_view) {
  return xla::cpu::EmitAtan2(module(), *b(), prim_type, lhs, rhs);
}

absl::StatusOr<toolchain::Value*> CpuElementalIrEmitter::EmitTanh(
    PrimitiveType prim_type, toolchain::Value* value) {
  if (prim_type == F32 || prim_type == F64 || prim_type == F16) {
    toolchain::Function* tanh =
        xla::codegen::intrinsics::Tanh::GetOrInsertDeclaration(
            module(), Type::S(prim_type));
    return b()->CreateCall(tanh, value);
  }
  return xla::cpu::EmitTanh(module(), *b(), prim_type, value);
}

absl::StatusOr<toolchain::Value*> CpuElementalIrEmitter::EmitErf(
    PrimitiveType prim_type, toolchain::Value* value) {
  return xla::cpu::EmitErf(module(), *b(), prim_type, value);
}

absl::StatusOr<toolchain::Value*> CpuElementalIrEmitter::EmitExp(
    PrimitiveType prim_type, toolchain::Value* value, absl::string_view name) {
  if (prim_type == F64) {
    toolchain::Type* f64 = b()->getDoubleTy();
    toolchain::Function* exp_f64 =
        xla::codegen::intrinsics::Exp::GetOrInsertDeclaration(
            module(), Type::TypeFromIrType(f64));
    return b()->CreateCall(exp_f64, value);
  }
  return llvm_ir::EmitCallToIntrinsic(toolchain::Intrinsic::exp, {value},
                                      {value->getType()}, b(), name);
}

absl::StatusOr<toolchain::Value*> CpuElementalIrEmitter::EmitRsqrt(
    PrimitiveType prim_type, toolchain::Value* value) {
  if (prim_type == F32 || prim_type == F64) {
    toolchain::Function* rsqrt_fn =
        xla::codegen::intrinsics::Rsqrt::GetOrInsertDeclaration(
            module(), Type::S(prim_type));
    return b()->CreateCall(rsqrt_fn, value);
  }
  toolchain::CallInst* sqrt = llvm_ir::EmitCallToIntrinsic(
      toolchain::Intrinsic::sqrt, {value}, {value->getType()}, b());
  return FDiv(toolchain::ConstantFP::get(sqrt->getType(), 1.0), sqrt);
}

absl::StatusOr<std::vector<toolchain::Value*>>
CpuElementalIrEmitter::EmitThreadLocalCall(
    const HloComputation& callee, absl::Span<toolchain::Value* const> parameters,
    absl::string_view name, bool is_reducer) {
  if (thread_local_call_fn_ == nullptr) {
    return absl::InternalError("Thread local call function is not set.");
  }

  return thread_local_call_fn_(callee, parameters, name, is_reducer);
}

}  // namespace xla::cpu
