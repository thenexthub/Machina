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

#if defined(INTEL_MKL)

#include "machina/xla/service/cpu/onednn_memory_util.h"

#include <string>

#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/IR/Argument.h"
#include "toolchain/IR/BasicBlock.h"
#include "toolchain/IR/DerivedTypes.h"
#include "toolchain/IR/Function.h"
#include "toolchain/IR/IRBuilder.h"
#include "toolchain/IR/Type.h"
#include "toolchain/IR/Value.h"
#include "machina/xla/hlo/testlib/filecheck.h"
#include "machina/xla/service/llvm_ir/llvm_util.h"
#include "machina/xla/shape.h"
#include "machina/xla/shape_util.h"
#include "machina/xla/tsl/lib/core/status_test_util.h"
#include "machina/xla/tsl/platform/test.h"

namespace xla {
namespace cpu {
namespace {

class MemoryUtilTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<std::vector<int64_t>> {
 protected:
  constexpr static const char* test_pattern_ = R"(
    CHECK: %[[mref0:[0-9]+]] = insertvalue
    CHECK: %[[mref1:[0-9]+]] = insertvalue
    CHECK-SAME: [[arr:\[12 x i64\]]] } %[[mref0]], i64 255, 3
    CHECK: %{{[0-9]+}} = insertvalue
    CHECK-SAME: %[[mref1]], [[arr]] )";

  auto GetMemRefTestPattern(Shape shape) {
    std::ostringstream stream;
    stream << "[";
    absl::c_for_each(shape.dimensions(),
                     [&stream](auto x) { stream << "i64 " << x << ", "; });
    return absl::StrCat(test_pattern_, stream.str());
  }
};

TEST_P(MemoryUtilTest, VerifyMemRefTest) {
  std::string filecheck_input;
  toolchain::LLVMContext context = toolchain::LLVMContext();
  toolchain::IRBuilder builder(context);
  toolchain::raw_string_ostream ostream(filecheck_input);
  toolchain::Module module("MemoryUtilTest", context);

  toolchain::FunctionType* function_type = toolchain::FunctionType::get(
      toolchain::Type::getVoidTy(context), {builder.getPtrTy()}, false);
  toolchain::Function* function = toolchain::Function::Create(
      function_type, toolchain::Function::LinkageTypes::ExternalLinkage,
      "memory_util_test", module);
  toolchain::BasicBlock* bb = toolchain::BasicBlock::Create(context, "BB", function);
  builder.SetInsertPoint(bb);

  Shape shape = ShapeUtil::MakeShape(F32, GetParam());
  toolchain::Argument* ptr = function->getArg(0);
  toolchain::Type* type = llvm_ir::PrimitiveTypeToIrType(F32, builder.getContext());

  if (shape.IsArray()) {
    for (auto dim : LayoutUtil::MinorToMajor(shape)) {
      type = toolchain::ArrayType::get(type, shape.dimensions(dim));
    }
  }

  llvm_ir::IrArray ir_array(ptr, type, shape);
  auto alloca = GetAllocaAndEmitMemrefInfo(builder, ir_array);
  alloca.EmitLifetimeEnd();
  ostream << module;

  absl::StatusOr<bool> match =
      RunFileCheck(filecheck_input, GetMemRefTestPattern(shape));
  TF_ASSERT_OK(match.status());
  EXPECT_TRUE(match.value());
}

INSTANTIATE_TEST_SUITE_P(
    MemoryUtilTestSuite, MemoryUtilTest,
    ::testing::Values(std::vector<int64_t>({30}),
                      std::vector<int64_t>({30, 40}),
                      std::vector<int64_t>({30, 40, 50})),
    [](const ::testing::TestParamInfo<MemoryUtilTest::ParamType>& info) {
      return absl::StrCat("Rank_", info.param.size());
    });

}  // namespace
}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL
