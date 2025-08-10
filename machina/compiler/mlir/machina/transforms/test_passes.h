/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_TEST_PASSES_H_
#define MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_TEST_PASSES_H_

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain

namespace mlir {
namespace tf_test {

// Returns test pass for variable freezing.
std::unique_ptr<OperationPass<ModuleOp>> CreateFreezeVariableTestPass();

// Test pass for applying TF->TF lowering patterns.
std::unique_ptr<OperationPass<func::FuncOp>> CreateTestTFLowerTFPass();

// Test passes for visitor util.
std::unique_ptr<OperationPass<func::FuncOp>> CreateTestVisitorUtilPass();
std::unique_ptr<OperationPass<func::FuncOp>>
CreateTestVisitorUtilInterruptPass();

// Test operation clustering based on user defined policy.
std::unique_ptr<OperationPass<func::FuncOp>> CreateTestClusteringPolicyPass();

// Test pass for analyzing side-effect analysis result.
std::unique_ptr<OperationPass<ModuleOp>> CreateTestSideEffectAnalysisPass();

std::unique_ptr<OperationPass<ModuleOp>> CreateTestResourceAliasAnalysisPass();

std::unique_ptr<OperationPass<ModuleOp>> CreateInitTextFileToImportTestPass();
std::unique_ptr<OperationPass<ModuleOp>>
CreateInitTextFileToImportSavedModelTestPass();

// Variable Lifting test passes: only useful for lit testing.
std::unique_ptr<OperationPass<ModuleOp>> CreateLiftVariablesTestPass();
std::unique_ptr<OperationPass<ModuleOp>>
CreateLiftVariablesInvalidSessionTestPass();

// Create a test pass for the above with a "fake" session, for lit testing.
std::unique_ptr<OperationPass<ModuleOp>>
CreateInitializeVariablesInSessionInitializerTestPass();

// Create a test pass that emits remarks for each analysis result for resources.
// This pass is only used for lit testing.
std::unique_ptr<OperationPass<ModuleOp>> CreateResourceAnalyzerTestPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_FREEZEVARIABLESTESTPASS
#define GEN_PASS_DECL_INITTEXTFILETOIMPORTSAVEDMODELTESTPASS
#define GEN_PASS_DECL_INITTEXTFILETOIMPORTTESTPASS
#define GEN_PASS_DECL_INITIALIZEVARIABLESINSESSIONINITIALIZERPASS
#define GEN_PASS_DECL_LIFTVARIABLESINVALIDSESSIONTESTPASS
#define GEN_PASS_DECL_LIFTVARIABLESTESTPASS
#define GEN_PASS_DECL_RESOURCEANALYZERTESTPASS
#define GEN_PASS_DECL_TESTCLUSTERINGPOLICYPASS
#define GEN_PASS_DECL_TESTRESOURCEALIASANALYSIS
#define GEN_PASS_DECL_TESTSIDEEFFECTANALYSISPASS
#define GEN_PASS_DECL_TESTMACHINALOWERTFPASS
#include "machina/compiler/mlir/machina/transforms/test_passes.h.inc"

}  // namespace tf_test
}  // namespace mlir
#endif  // MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_TEST_PASSES_H_
