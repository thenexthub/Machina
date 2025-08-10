/* Copyright 2022 The OpenXLA Authors.

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

#include "machina/xla/service/gpu/target_util.h"

#include <gtest/gtest.h>
#include "toolchain/IR/BasicBlock.h"
#include "toolchain/IR/DerivedTypes.h"
#include "toolchain/IR/Function.h"
#include "toolchain/IR/IRBuilder.h"
#include "toolchain/IR/LLVMContext.h"
#include "toolchain/IR/Verifier.h"
#include "toolchain/Support/raw_ostream.h"
#include "toolchain/TargetParser/Triple.h"
#include "machina/xla/xla_data.pb.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class TargetUtilTest : public testing::Test {
 public:
  TargetUtilTest() : module_("test", ctx_), builder_(ctx_) {}

 protected:
  void SetUp() override {
    auto fn = toolchain::Function::Create(
        toolchain::FunctionType::get(toolchain::Type::getVoidTy(ctx_), {}),
        toolchain::Function::LinkageTypes::ExternalLinkage, "fn", module_);
    auto block = toolchain::BasicBlock::Create(ctx_, "blk", fn);
    builder_.SetInsertPoint(block);
  }

  toolchain::LLVMContext ctx_;
  toolchain::Module module_;
  toolchain::IRBuilder<> builder_;
};

TEST_F(TargetUtilTest, NVPTXGroupBarrier) {
  module_.setTargetTriple(toolchain::Triple("nvptx"));
  EmitCallToTargetIntrinsic(TargetIntrinsicID::kGroupBarrierId,
                            {/*membermask=*/builder_.getInt32(-1)}, {},
                            &builder_);
  builder_.CreateRetVoid();
  EXPECT_FALSE(toolchain::verifyModule(module_, &toolchain::errs()));
}

TEST_F(TargetUtilTest, AMDGCNGroupBarrier) {
  module_.setTargetTriple(toolchain::Triple("amdgcn"));
  EmitCallToTargetIntrinsic(TargetIntrinsicID::kGroupBarrierId, {}, {},
                            &builder_);
  builder_.CreateRetVoid();
  EXPECT_FALSE(toolchain::verifyModule(module_, &toolchain::errs()));
}

TEST(TargetUtil, ObtainDeviceFunctionNameExp) {
  toolchain::Triple triple("nvptx64-unknown-unknown");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kExp,
                                     /*output_type=*/F32, triple),
            "__nv_expf");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kExp,
                                     /*output_type=*/BF16, triple),
            "__nv_fast_expf");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kExp,
                                     /*output_type=*/F16, triple),
            "__nv_fast_expf");
}

TEST(TargetUtil, ObtainDeviceFunctionNameLog) {
  toolchain::Triple triple("nvptx64-unknown-unknown");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kLog,
                                     /*output_type=*/F32, triple),
            "__nv_logf");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kLog,
                                     /*output_type=*/BF16, triple),
            "__nv_fast_logf");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kLog,
                                     /*output_type=*/F16, triple),
            "__nv_fast_logf");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
