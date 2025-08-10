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

#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "toolchain/Analysis/CGSCCPassManager.h"
#include "toolchain/Analysis/LoopAnalysisManager.h"
#include "toolchain/IR/Constant.h"
#include "toolchain/IR/Constants.h"
#include "toolchain/IR/DerivedTypes.h"
#include "toolchain/IR/GlobalValue.h"
#include "toolchain/IR/GlobalVariable.h"
#include "toolchain/IR/Module.h"
#include "toolchain/IR/PassManager.h"
#include "toolchain/Passes/PassBuilder.h"
#include "toolchain/Support/Casting.h"
#include "toolchain/Transforms/IPO/AlwaysInliner.h"
#include "toolchain/Transforms/IPO/GlobalDCE.h"
#include "toolchain/Transforms/InstCombine/InstCombine.h"
#include "toolchain/Transforms/Scalar/DCE.h"
#include "toolchain/Transforms/Scalar/EarlyCSE.h"

namespace xla::codegen::intrinsic {

void RunInlineAndOptPasses(toolchain::Module& module) {
  toolchain::PassBuilder pb;

  toolchain::LoopAnalysisManager lam;
  toolchain::FunctionAnalysisManager fam;
  toolchain::CGSCCAnalysisManager cgam;
  toolchain::ModuleAnalysisManager mam;

  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  toolchain::ModulePassManager mpm;
  mpm.addPass(toolchain::AlwaysInlinerPass());

  toolchain::FunctionPassManager fpm;
  fpm.addPass(toolchain::InstCombinePass());
  fpm.addPass(toolchain::EarlyCSEPass());
  fpm.addPass(toolchain::DCEPass());
  mpm.addPass(toolchain::createModuleToFunctionPassAdaptor(std::move(fpm)));
  mpm.addPass(toolchain::GlobalDCEPass());

  mpm.run(module, mam);
}

constexpr absl::string_view kCompilerUsedName = "toolchain.compiler.used";

void RemoveFromCompilerUsed(
    toolchain::Module& module,
    absl::flat_hash_set<absl::string_view> replaced_functions) {
  if (replaced_functions.empty()) {
    return;
  }

  toolchain::GlobalVariable* compiler_used =
      module.getNamedGlobal(kCompilerUsedName);
  if (!compiler_used) {
    return;
  }

  toolchain::ConstantArray* old_array =
      toolchain::dyn_cast<toolchain::ConstantArray>(compiler_used->getInitializer());
  if (!old_array) {
    return;
  }

  // Collect the constants that should be kept.
  std::vector<toolchain::Constant*> elements;
  elements.reserve(old_array->getNumOperands());
  for (int i = 0; i < old_array->getNumOperands(); ++i) {
    auto* operand = old_array->getOperand(i);
    toolchain::GlobalValue* gv =
        toolchain::dyn_cast<toolchain::GlobalValue>(operand->stripPointerCasts());

    if (gv && replaced_functions.contains(gv->getName())) {
      continue;
    }
    elements.push_back(operand);
  }

  // If all functions were removed, erase the global entirely.
  if (elements.empty()) {
    compiler_used->eraseFromParent();
    return;
  }

  // If only some functions were removed, modify the existing global in-place.
  if (elements.size() < old_array->getNumOperands()) {
    toolchain::ArrayType* new_array_type = toolchain::ArrayType::get(
        old_array->getType()->getElementType(), elements.size());
    toolchain::Constant* new_array_init =
        toolchain::ConstantArray::get(new_array_type, elements);

    // Create a new global toolchain.compiler.used with the new contents.
    auto new_global =
        new toolchain::GlobalVariable(module, new_array_type, false,
                                 compiler_used->getLinkage(), new_array_init);
    new_global->copyAttributesFrom(compiler_used);
    new_global->setSection(compiler_used->getSection());
    new_global->setAlignment(compiler_used->getAlign());
    new_global->takeName(compiler_used);

    compiler_used->eraseFromParent();
  }
}

}  // namespace xla::codegen::intrinsic
