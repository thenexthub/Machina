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

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "toolchain/ADT/StringMap.h"
#include "toolchain/ExecutionEngine/Orc/CompileUtils.h"
#include "toolchain/IR/BasicBlock.h"
#include "toolchain/IR/Constants.h"
#include "toolchain/IR/DerivedTypes.h"
#include "toolchain/IR/Function.h"
#include "toolchain/IR/GlobalValue.h"
#include "toolchain/IR/IRBuilder.h"
#include "toolchain/IR/Intrinsics.h"
#include "toolchain/IR/Verifier.h"
#include "toolchain/TargetParser/Host.h"
#include "machina/xla/codegen/intrinsic/intrinsic.h"
#include "machina/xla/codegen/intrinsic/simple_jit_runner.h"
#include "machina/xla/codegen/intrinsic/test_matchers.h"
#include "machina/xla/primitive_util.h"
#include "machina/xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {
namespace {

using ::xla::codegen::intrinsic::JitRunner;
using ::xla::codegen::intrinsic::NearUlps;

TEST(RsqrtTest, Name) {
  EXPECT_EQ(Rsqrt::Name(Type::S(F32)), "xla.rsqrt.f32");
  EXPECT_EQ(Rsqrt::Name(Type::V(F32, 4)), "xla.rsqrt.v4f32");
  EXPECT_EQ(Rsqrt::Name(Type::V(F64, 8)), "xla.rsqrt.v8f64");
}

constexpr int kF32UlpsPrecision = 1;
constexpr int kF64UlpsPrecision = 1;

void AddOneOverSqrt(toolchain::LLVMContext& context, toolchain::Module& module,
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
  EXPECT_FALSE(toolchain::verifyFunction(*rsqrt_func));

  AddOneOverSqrt(*context, *module, rsqrt_func->getReturnType());
  return JitRunner(std::move(module), std::move(context));
}

bool hasAvx() {
  toolchain::StringMap<bool> HostFeatures = toolchain::sys::getHostCPUFeatures();
  return HostFeatures.lookup("avx");
}

bool hasAvx512Support() {
  toolchain::StringMap<bool> HostFeatures = toolchain::sys::getHostCPUFeatures();
  return HostFeatures.lookup("avx512f");
}

TEST(FeaturesTest, HostFeatures) {
  std::cout << "Host features x86:" << hasAvx()
            << ", avx512f:" << hasAvx512Support() << "\n";
}

TEST(RsqrtTest, EmitRsqrtF32) {
  if (hasAvx()) {
    Type type = Type::S(F32);
    JitRunner jit = CreateJitRunnerWithRsqrt(type);
    auto rsqrt = jit.GetScalarFn<float(float)>(Rsqrt::Name(type));
    auto one_over_sqrt = jit.GetScalarFn<float(float)>("one_over_sqrt");
    float vals[] = {
        1.0f,
        4.0f,
        0.25f,
        100.0f,
        1e-10f,
        1e10f,
        std::numeric_limits<float>::min(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::infinity(),
        -1.0f,  // Should produce NaN
        0.0f,   // Should produce infinity
        std::numeric_limits<float>::quiet_NaN(),
    };

    for (float val : vals) {
      float actual = rsqrt(val);
      float expected = one_over_sqrt(val);

      if (std::isnan(expected)) {
        EXPECT_TRUE(std::isnan(actual)) << "val = " << val;
      } else if (std::isinf(expected)) {
        EXPECT_TRUE(std::isinf(actual)) << "val = " << val;
        EXPECT_EQ(expected > 0, actual > 0) << "val = " << val;
      } else {
        EXPECT_THAT(actual, NearUlps<float>(expected, kF32UlpsPrecision))
            << "val = " << val;
      }
    }
  }
}

template <size_t kN, PrimitiveType prim_type>
void TestRsqrt_Vectors() {
  Type type = Type::V(prim_type, kN);
  JitRunner jit = CreateJitRunnerWithRsqrt(type);
  using NativeType = primitive_util::NativeTypeOf<prim_type>;
  auto rsqrt =
      jit.GetVectorizedFn<kN, NativeType, NativeType>(Rsqrt::Name(type));
  std::vector<NativeType> val_vec = {1.0f, 0.0f, 0.25f, 100.0f, -1.0f};
  std::array<NativeType, kN> vals;
  for (size_t i = 0; i < kN; ++i) {
    vals[i] = val_vec[i % val_vec.size()];
  }
  std::array<NativeType, kN> actuals = rsqrt(vals);

  size_t prec = prim_type == F32 ? kF32UlpsPrecision : kF64UlpsPrecision;
  for (int i = 0; i < kN; ++i) {
    NativeType expected = 1.0f / std::sqrt(vals[i]);
    EXPECT_THAT(actuals[i], NearUlps<NativeType>(expected, prec))
        << "i = " << i << " val = " << vals[i] << " kN= " << kN;
  }
}

TEST(RsqrtTest, EmitRsqrtF32_Vectors) {
  if (hasAvx()) {
    TestRsqrt_Vectors<4, F32>();
    TestRsqrt_Vectors<8, F32>();
    if (hasAvx512Support()) {
      TestRsqrt_Vectors<16, F32>();
    }
  }
}

TEST(RsqrtTest, EmitRsqrtF64_Vectors) {
  if (hasAvx512Support()) {
    TestRsqrt_Vectors<2, F64>();
    TestRsqrt_Vectors<4, F64>();
    TestRsqrt_Vectors<8, F64>();
  }
}

TEST(RsqrtTest, EmitRsqrtF32_EdgeCases) {
  if (hasAvx()) {
    Type type = Type::S(F32);
    JitRunner jit = CreateJitRunnerWithRsqrt(type);
    auto rsqrt = jit.GetScalarFn<float(float)>(Rsqrt::Name(type));

    float actual_denorm = rsqrt(std::numeric_limits<float>::denorm_min());
    EXPECT_THAT(actual_denorm,
                NearUlps<float>(std::numeric_limits<float>::infinity(),
                                kF32UlpsPrecision));

    float large_val = std::numeric_limits<float>::max();
    float actual_large = rsqrt(large_val);
    float expected_large = 1.0f / std::sqrt(large_val);
    EXPECT_THAT(actual_large,
                NearUlps<float>(expected_large, kF32UlpsPrecision));

    float small_val = std::numeric_limits<float>::min();
    float actual_small = rsqrt(small_val);
    float expected_small = 1.0f / std::sqrt(small_val);
    EXPECT_THAT(actual_small,
                NearUlps<float>(expected_small, kF32UlpsPrecision));
  }
}

TEST(RsqrtTest, EmitRsqrtF64) {
  if (hasAvx()) {
    Type type = Type::S(F64);
    JitRunner jit = CreateJitRunnerWithRsqrt(type);
    auto rsqrt = jit.GetScalarFn<double(double)>(Rsqrt::Name(type));
    auto one_over_sqrt = jit.GetScalarFn<double(double)>("one_over_sqrt");

    EXPECT_THAT(rsqrt(1234.0),
                NearUlps<double>(one_over_sqrt(1234.0), kF64UlpsPrecision));
  }
}

TEST(RsqrtTest, EmitRsqrtF64_EdgeCasesAvxFallback) {
  if (hasAvx() && !hasAvx512Support()) {
    Type type = Type::S(F64);
    JitRunner jit = CreateJitRunnerWithRsqrt(type);
    auto rsqrt = jit.GetScalarFn<double(double)>(Rsqrt::Name(type));
    EXPECT_THAT(rsqrt(std::numeric_limits<double>::infinity()),
                NearUlps<double>(0.0, kF64UlpsPrecision));

    // NB: The fallback 1/ sqrt(x) doesn't return 0 for max double.
    // EXPECT_THAT(rsqrt(std::numeric_limits<double>::max()),
    //             NearUlps<double>(0.0, kF64UlpsPrecision));
  }
}

TEST(RsqrtTest, EmitRsqrtF64_EdgeCasesHasAvx) {
  if (hasAvx512Support()) {
    Type type = Type::S(F64);
    JitRunner jit = CreateJitRunnerWithRsqrt(type);
    auto rsqrt = jit.GetScalarFn<double(double)>(Rsqrt::Name(type));
    auto one_over_sqrt = jit.GetScalarFn<double(double)>("one_over_sqrt");
    EXPECT_THAT(rsqrt(std::numeric_limits<double>::infinity()),
                NearUlps<double>(0.0, kF64UlpsPrecision));
    double max = std::numeric_limits<double>::max();
    EXPECT_THAT(rsqrt(max),
                NearUlps<double>(one_over_sqrt(max), kF64UlpsPrecision));
    double large = 8.5390423905955551e+307;
    EXPECT_THAT(rsqrt(large),
                NearUlps<double>(one_over_sqrt(large), kF64UlpsPrecision));
    double large2 = 6.112156648698989e+307;
    EXPECT_THAT(rsqrt(large2),
                NearUlps<double>(one_over_sqrt(large2), kF64UlpsPrecision));
  }
}

template <size_t kN>
void TestRsqrtF64EdgeCases() {
  if (hasAvx512Support()) {
    Type type = Type::V(F64, kN);
    JitRunner jit = CreateJitRunnerWithRsqrt(type);
    using NativeType = double;
    auto rsqrt =
        jit.GetVectorizedFn<kN, NativeType, NativeType>(Rsqrt::Name(type));
    auto one_over_sqrt =
        jit.GetVectorizedFn<kN, NativeType, NativeType>("one_over_sqrt");
    std::vector<NativeType> val_vec = {
        std::numeric_limits<double>::denorm_min(),
        std::numeric_limits<double>::max()};
    std::array<NativeType, kN> vals;
    for (size_t i = 0; i < kN; ++i) {
      vals[i] = val_vec[i % val_vec.size()];
    }
    std::array<NativeType, kN> actuals = rsqrt(vals);
    std::array<NativeType, kN> expected = one_over_sqrt(vals);
    for (int i = 0; i < kN; ++i) {
      EXPECT_THAT(actuals[i],
                  NearUlps<NativeType>(expected[i], kF64UlpsPrecision))
          << "i = " << i << " val = " << vals[i] << " kN= " << kN;
    }

    std::array<NativeType, kN> map_to_tiny = {
        8.5390423905955551e+307, std::numeric_limits<double>::infinity()};
    std::array<NativeType, kN> map_to_tiny_vals;
    for (size_t i = 0; i < kN; ++i) {
      map_to_tiny_vals[i] = map_to_tiny[i % map_to_tiny.size()];
    }
    std::array<NativeType, kN> actual_tiny = rsqrt(map_to_tiny_vals);
    std::array<NativeType, kN> expected_tiny = one_over_sqrt(map_to_tiny_vals);
    for (size_t i = 0; i < kN; ++i) {
      EXPECT_THAT(actual_tiny[i],
                  NearUlps<NativeType>(expected_tiny[i], kF64UlpsPrecision))
          << "i = " << i << " val = " << map_to_tiny_vals[i] << " kN= " << kN;
    }

    // Test a value that is close to the edge of the range where the refinement
    // is not used.
    std::array<NativeType, kN> edge_vals;
    for (size_t i = 0; i < kN; ++i) {
      edge_vals[i] = 4.5e+307;
    }
    std::array<NativeType, kN> actual_edge = rsqrt(edge_vals);
    std::array<NativeType, kN> expected_edge = one_over_sqrt(edge_vals);
    for (size_t i = 0; i < kN; ++i) {
      EXPECT_THAT(actual_edge[i],
                  NearUlps<NativeType>(expected_edge[i], kF64UlpsPrecision))
          << "i = " << i << " val = " << edge_vals[i] << " kN= " << kN;
    }
  }
}

TEST(RsqrtTest, EmitRsqrtF64_EdgeCases_Vectors) {
  if (hasAvx512Support()) {
    TestRsqrtF64EdgeCases<2>();
    TestRsqrtF64EdgeCases<4>();
    TestRsqrtF64EdgeCases<8>();
  }
}

}  // namespace
}  // namespace xla::codegen::intrinsics
