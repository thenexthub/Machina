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

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "toolchain/IR/BasicBlock.h"
#include "toolchain/IR/Constants.h"
#include "toolchain/IR/DerivedTypes.h"
#include "toolchain/IR/Function.h"
#include "toolchain/IR/GlobalValue.h"
#include "toolchain/IR/IRBuilder.h"
#include "toolchain/IR/Intrinsics.h"
#include "toolchain/IR/LLVMContext.h"
#include "toolchain/IR/Module.h"
#include "toolchain/IR/Type.h"
#include "toolchain/IR/Value.h"
#include "toolchain/IR/Verifier.h"
#include "toolchain/Target/TargetMachine.h"
#include "machina/xla/codegen/intrinsic/intrinsic.h"
#include "machina/xla/codegen/intrinsic/rsqrt.h"
#include "machina/xla/codegen/intrinsic/simple_jit_runner.h"
#include "machina/xla/primitive_util.h"
#include "machina/xla/tsl/platform/test_benchmark.h"
#include "machina/xla/xla_data.pb.h"

namespace xla::codegen::intrinsic {

using ::xla::codegen::intrinsics::Rsqrt;
using ::xla::codegen::intrinsics::Type;

void CreateOneOverSqrt(toolchain::LLVMContext& context, toolchain::Module& module,
                       toolchain::Type* type) {
  // insert 1 / sqrt(x) function for comparison.
  toolchain::Function* one_over_sqrt_func = toolchain::Function::Create(
      toolchain::FunctionType::get(type, {type}, /*isVarArg=*/false),
      toolchain::GlobalValue::ExternalLinkage, "one_over_sqrt", module);
  toolchain::BasicBlock* entry_bb =
      toolchain::BasicBlock::Create(context, "entry", one_over_sqrt_func);
  toolchain::Value* x = one_over_sqrt_func->getArg(0);
  toolchain::IRBuilder<> builder(entry_bb);
  toolchain::Value* one_over_sqrt = builder.CreateFDiv(
      toolchain::ConstantFP::get(type, 1.0),
      builder.CreateUnaryIntrinsic(toolchain::Intrinsic::sqrt, x));
  builder.CreateRet(one_over_sqrt);
}

JitRunner CreateJitRunnerWithRsqrt(Type type) {
  auto context = std::make_unique<toolchain::LLVMContext>();
  auto module = std::make_unique<toolchain::Module>("test_module", *context);
  std::unique_ptr<toolchain::TargetMachine> target_machine =
      xla::codegen::intrinsic::CreateHostTargetMachine();
  toolchain::Function* rsqrt_func =
      Rsqrt::CreateDefinition(
          module.get(), target_machine->getTargetFeatureString().str(), type)
          .value();
  rsqrt_func->setLinkage(toolchain::Function::ExternalLinkage);
  CreateOneOverSqrt(*context, *module, Type::TypeToIrType(type, *context));
  return JitRunner(std::move(module), std::move(context));
}

enum RsqrtFunction {
  kRsqrt,
  kOneOverSqrt,
};

template <size_t num_elements, PrimitiveType type, RsqrtFunction function>
static void BM_RsqrtVectorized(benchmark::State& state) {
  using NativeType = typename primitive_util::PrimitiveTypeToNative<type>::type;
  Type intrinsic_type = Type::V(type, num_elements);
  JitRunner jit = CreateJitRunnerWithRsqrt(intrinsic_type);
  std::string function_name =
      (function == kRsqrt) ? Rsqrt::Name(intrinsic_type) : "one_over_sqrt";
  auto rsqrt = jit.GetVectorizedFn<num_elements, NativeType, NativeType>(
      function_name, 100'000);
  std::array<NativeType, num_elements> vec;
  for (size_t i = 0; i < num_elements; ++i) {
    vec[i] = static_cast<NativeType>(100.0 + i * 10.0);
  }
  for (auto s : state) {
    rsqrt(vec);
  }
}

BENCHMARK(BM_RsqrtVectorized<4, F32, kRsqrt>)->MeasureProcessCPUTime();
BENCHMARK(BM_RsqrtVectorized<4, F32, kOneOverSqrt>)->MeasureProcessCPUTime();
BENCHMARK(BM_RsqrtVectorized<8, F32, kRsqrt>)->MeasureProcessCPUTime();
BENCHMARK(BM_RsqrtVectorized<8, F32, kOneOverSqrt>)->MeasureProcessCPUTime();
BENCHMARK(BM_RsqrtVectorized<2, F64, kRsqrt>)->MeasureProcessCPUTime();
BENCHMARK(BM_RsqrtVectorized<2, F64, kOneOverSqrt>)->MeasureProcessCPUTime();
BENCHMARK(BM_RsqrtVectorized<4, F64, kRsqrt>)->MeasureProcessCPUTime();
BENCHMARK(BM_RsqrtVectorized<4, F64, kOneOverSqrt>)->MeasureProcessCPUTime();
BENCHMARK(BM_RsqrtVectorized<8, F64, kRsqrt>)->MeasureProcessCPUTime();
BENCHMARK(BM_RsqrtVectorized<8, F64, kOneOverSqrt>)->MeasureProcessCPUTime();
}  // namespace xla::codegen::intrinsic
