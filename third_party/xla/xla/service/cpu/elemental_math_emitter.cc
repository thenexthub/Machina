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

#include "machina/xla/service/cpu/elemental_math_emitter.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "toolchain/IR/CallingConv.h"
#include "toolchain/IR/IRBuilder.h"
#include "toolchain/IR/Value.h"
#include "toolchain/Support/Casting.h"
#include "machina/xla/codegen/intrinsic/erf.h"
#include "machina/xla/codegen/intrinsic/intrinsic.h"
#include "machina/xla/xla_data.pb.h"

namespace xla::cpu {

using ::xla::codegen::intrinsics::Type;

absl::StatusOr<toolchain::Value*> EmitAtan2(toolchain::Module* module,
                                       toolchain::IRBuilderBase& b,
                                       PrimitiveType prim_type,
                                       toolchain::Value* lhs, toolchain::Value* rhs) {
  std::string function_name;
  bool cast_result_to_fp16 = false;
  switch (prim_type) {
    case F16:
      cast_result_to_fp16 = true;
      lhs = b.CreateFPCast(lhs, b.getFloatTy());
      rhs = b.CreateFPCast(rhs, b.getFloatTy());
      [[fallthrough]];
    case F32:
      function_name = "atan2f";
      break;
    case F64:
      function_name = "atan2";
      break;
    default:
      return absl::UnimplementedError("atan2");
  }
  // Create a function declaration.
  toolchain::Function* function = toolchain::dyn_cast<toolchain::Function>(
      module
          ->getOrInsertFunction(function_name, lhs->getType(), lhs->getType(),
                                rhs->getType())
          .getCallee());
  function->setCallingConv(toolchain::CallingConv::C);
  function->setDoesNotThrow();
  function->setDoesNotAccessMemory();
  // Create an instruction to call the function.
  toolchain::Value* result = b.CreateCall(function, {lhs, rhs});
  if (cast_result_to_fp16) {
    result = b.CreateFPCast(result, b.getHalfTy());
  }
  return result;
}

absl::StatusOr<toolchain::Value*> EmitTanh(toolchain::Module* module,
                                      toolchain::IRBuilderBase& b,
                                      PrimitiveType prim_type,
                                      toolchain::Value* value) {
  bool cast_result_to_fp16 = false;
  std::string function_name;
  switch (prim_type) {
    case F16:
      cast_result_to_fp16 = true;
      value = b.CreateFPCast(value, b.getFloatTy());
      [[fallthrough]];
    case F32:
      function_name = "tanhf";
      break;
    case F64:
      function_name = "tanh";
      break;
    default:
      return absl::UnimplementedError("tanh");
  }
  // Create a function declaration.
  toolchain::Function* function = toolchain::dyn_cast<toolchain::Function>(
      module
          ->getOrInsertFunction(function_name, value->getType(),
                                value->getType())
          .getCallee());
  function->setCallingConv(toolchain::CallingConv::C);
  function->setDoesNotThrow();
  function->setDoesNotAccessMemory();
  // Create an instruction to call the function.
  toolchain::Value* result = b.CreateCall(function, value);
  if (cast_result_to_fp16) {
    result = b.CreateFPCast(result, b.getHalfTy());
  }
  return result;
}

absl::StatusOr<toolchain::Value*> EmitErf(toolchain::Module* module,
                                     toolchain::IRBuilderBase& b,
                                     PrimitiveType prim_type,
                                     toolchain::Value* value) {
  if (prim_type == F64) {
    std::string function_name = "erf";
    // Create a function declaration.
    toolchain::Function* function = toolchain::dyn_cast<toolchain::Function>(
        module
            ->getOrInsertFunction(function_name, value->getType(),
                                  value->getType())
            .getCallee());
    function->setCallingConv(toolchain::CallingConv::C);
    function->setDoesNotThrow();
    function->setDoesNotAccessMemory();
    // Create an instruction to call the function.
    toolchain::Value* result = b.CreateCall(function, value);
    return result;
  }
  // Upcast F16 to F32 if necessary.
  toolchain::Type* type = prim_type == F16 ? b.getFloatTy() : value->getType();
  if (type == b.getFloatTy()) {
    toolchain::Value* x = b.CreateFPCast(value, type);
    toolchain::Function* erf =
        codegen::intrinsics::Erf::GetOrInsertDeclaration(module, Type::S(F32));
    toolchain::Value* result = b.CreateCall(erf, {x});
    return b.CreateFPCast(result, value->getType());
  }
  return absl::UnimplementedError("erf");
}

}  // namespace xla::cpu
