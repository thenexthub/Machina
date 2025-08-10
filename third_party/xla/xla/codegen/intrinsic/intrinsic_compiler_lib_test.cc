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

#include "machina/xla/codegen/intrinsic/intrinsic_compiler_lib.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "toolchain/IR/BasicBlock.h"
#include "toolchain/IR/CallingConv.h"
#include "toolchain/IR/Constants.h"
#include "toolchain/IR/DerivedTypes.h"
#include "toolchain/IR/Function.h"
#include "toolchain/IR/GlobalValue.h"
#include "toolchain/IR/GlobalVariable.h"
#include "toolchain/IR/Instructions.h"
#include "toolchain/IR/LLVMContext.h"
#include "toolchain/IR/Module.h"
#include "toolchain/IR/Type.h"
#include "toolchain/Support/Casting.h"

namespace xla::codegen::intrinsic {
namespace {

class RemoveFromCompilerUsedTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_ = std::make_unique<toolchain::LLVMContext>();
    module_ = std::make_unique<toolchain::Module>("test_module", *context_);
  }

  toolchain::Function* CreateTestFunction(const std::string& name) {
    toolchain::FunctionType* func_type =
        toolchain::FunctionType::get(toolchain::Type::getVoidTy(*context_), {}, false);
    return toolchain::Function::Create(func_type, toolchain::Function::ExternalLinkage,
                                  name, *module_);
  }

  void CreateCompilerUsedArray(const std::vector<std::string>& function_names) {
    std::vector<toolchain::Constant*> elements;

    for (const std::string& name : function_names) {
      toolchain::Function* func = CreateTestFunction(name);
      elements.push_back(func);
    }

    toolchain::ArrayType* array_type = toolchain::ArrayType::get(
        toolchain::PointerType::getUnqual(*context_), elements.size());
    toolchain::Constant* array_init = toolchain::ConstantArray::get(array_type, elements);

    new toolchain::GlobalVariable(*module_, array_type, false,
                             toolchain::GlobalValue::AppendingLinkage, array_init,
                             "toolchain.compiler.used");
  }

  std::vector<std::string> GetCompilerUsedFunctionNames() {
    toolchain::GlobalVariable* compiler_used =
        module_->getNamedGlobal("toolchain.compiler.used");
    if (!compiler_used) {
      return {};
    }

    toolchain::ConstantArray* array =
        toolchain::dyn_cast<toolchain::ConstantArray>(compiler_used->getInitializer());
    if (!array) {
      return {};
    }

    std::vector<std::string> names;
    for (unsigned i = 0; i < array->getNumOperands(); ++i) {
      toolchain::GlobalValue* gv = toolchain::dyn_cast<toolchain::GlobalValue>(
          array->getOperand(i)->stripPointerCasts());
      if (gv) {
        names.push_back(gv->getName().str());
      }
    }
    return names;
  }

  std::unique_ptr<toolchain::LLVMContext> context_;
  std::unique_ptr<toolchain::Module> module_;
};

TEST_F(RemoveFromCompilerUsedTest, RemovesSpecifiedFunctions) {
  CreateCompilerUsedArray({"func1", "func2", "func3", "func4"});
  absl::flat_hash_set<absl::string_view> to_remove = {"func2", "func4"};

  RemoveFromCompilerUsed(*module_, to_remove);

  std::vector<std::string> remaining = GetCompilerUsedFunctionNames();
  EXPECT_EQ(remaining.size(), 2) << absl::StrJoin(remaining, ", ");
  EXPECT_TRUE(std::find(remaining.begin(), remaining.end(), "func1") !=
              remaining.end());
  EXPECT_TRUE(std::find(remaining.begin(), remaining.end(), "func3") !=
              remaining.end());
}

TEST_F(RemoveFromCompilerUsedTest, RemovesEntireArrayWhenAllFunctionsRemoved) {
  CreateCompilerUsedArray({"func1", "func2"});
  absl::flat_hash_set<absl::string_view> to_remove = {"func1", "func2"};

  RemoveFromCompilerUsed(*module_, to_remove);

  EXPECT_EQ(module_->getNamedGlobal("toolchain.compiler.used"), nullptr);
}

TEST_F(RemoveFromCompilerUsedTest, HandlesNoCompilerUsedArray) {
  // Arrange - no @toolchain.compiler.used exists
  absl::flat_hash_set<absl::string_view> to_remove = {"func1"};

  // Act - should not crash
  RemoveFromCompilerUsed(*module_, to_remove);

  EXPECT_EQ(module_->getNamedGlobal("toolchain.compiler.used"), nullptr);
}

TEST_F(RemoveFromCompilerUsedTest, DoesNothingWhenNoMatches) {
  CreateCompilerUsedArray({"func1", "func2"});
  absl::flat_hash_set<absl::string_view> to_remove = {"nonexistent"};

  RemoveFromCompilerUsed(*module_, to_remove);

  std::vector<std::string> remaining = GetCompilerUsedFunctionNames();
  EXPECT_EQ(remaining.size(), 2);
  EXPECT_TRUE(std::find(remaining.begin(), remaining.end(), "func1") !=
              remaining.end());
  EXPECT_TRUE(std::find(remaining.begin(), remaining.end(), "func2") !=
              remaining.end());
}

TEST_F(RemoveFromCompilerUsedTest, HandlesEmptyRemovalSet) {
  CreateCompilerUsedArray({"func1", "func2"});
  absl::flat_hash_set<absl::string_view> to_remove = {};

  RemoveFromCompilerUsed(*module_, to_remove);

  std::vector<std::string> remaining = GetCompilerUsedFunctionNames();
  EXPECT_EQ(remaining.size(), 2);
  EXPECT_TRUE(std::find(remaining.begin(), remaining.end(), "func1") !=
              remaining.end());
  EXPECT_TRUE(std::find(remaining.begin(), remaining.end(), "func2") !=
              remaining.end());
}

TEST(MathCompilerLibTest, InlineAndOptPasses) {
  toolchain::LLVMContext context;
  toolchain::Module module("test", context);
  toolchain::Function* f = toolchain::Function::Create(
      toolchain::FunctionType::get(toolchain::Type::getInt32Ty(context), {}, false),
      toolchain::GlobalValue::InternalLinkage, "f", &module);
  f->setCallingConv(toolchain::CallingConv::Fast);
  toolchain::BasicBlock* entry = toolchain::BasicBlock::Create(context, "entry", f);
  toolchain::ReturnInst::Create(
      context, toolchain::ConstantInt::get(toolchain::Type::getInt32Ty(context), 42),
      entry);

  RunInlineAndOptPasses(module);

  EXPECT_TRUE(module.getNamedGlobal("f") == nullptr);
}

}  // namespace
}  // namespace xla::codegen::intrinsic
