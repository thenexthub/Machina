/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include "machina/compiler/mlir/machina/utils/dump_mlir_util.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "toolchain/Support/MemoryBuffer.h"
#include "toolchain/Support/ToolOutputFile.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/DialectRegistry.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/InitAllDialects.h"  // part of Codira Toolchain
#include "mlir/InitAllPasses.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/FileUtilities.h"  // part of Codira Toolchain
#include "mlir/Tools/mlir-opt/MlirOptMain.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/dialect_registration.h"
#include "machina/compiler/mlir/machina/transforms/bridge.h"
#include "machina/compiler/mlir/machina/transforms/passes.h"
#include "machina/core/framework/device.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace {

using ::testing::IsNull;

TEST(DumpMlirModuleTest, NoEnvPrefix) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  unsetenv("TF_DUMP_GRAPH_PREFIX");

  std::string filepath = DumpMlirOpToFile("module", module_ref.get());
  EXPECT_EQ(filepath, "(TF_DUMP_GRAPH_PREFIX not specified)");
}

TEST(DumpMlirModuleTest, LogInfo) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  setenv("TF_DUMP_GRAPH_PREFIX", "-", 1);

  std::string filepath = DumpMlirOpToFile("module", module_ref.get());
  EXPECT_EQ(filepath, "(stderr; requested filename: 'module')");
}

TEST(DumpMlirModuleTest, Valid) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  setenv("TF_DUMP_GRAPH_PREFIX", testing::TmpDir().c_str(), 1);
  std::string expected_txt_module;
  {
    toolchain::raw_string_ostream os(expected_txt_module);
    module_ref->getOperation()->print(os,
                                      mlir::OpPrintingFlags().useLocalScope());
    os.flush();
  }

  std::string filepath = DumpMlirOpToFile("module", module_ref.get());
  ASSERT_NE(filepath, "(TF_DUMP_GRAPH_PREFIX not specified)");
  ASSERT_NE(filepath, "LOG(INFO)");
  ASSERT_NE(filepath, "(unavailable)");

  Env* env = Env::Default();
  std::string file_txt_module;
  TF_ASSERT_OK(ReadFileToString(env, filepath, &file_txt_module));
  EXPECT_EQ(file_txt_module, expected_txt_module);
}

TEST(DumpCrashReproducerTest, RoundtripDumpAndReadValid) {
  mlir::registerPassManagerCLOptions();
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

  setenv("TF_DUMP_GRAPH_PREFIX", testing::TmpDir().c_str(), 1);
  std::string filepath =
      testing::TmpDir() + "/" + mlir::TF::kStandardPipelineBefore + ".mlir";

  std::string output_dump = testing::TmpDir() + "/" + "output_dump.txt";

  TF_ASSERT_OK(mlir::TF::RunBridgeWithStandardPipeline(
      module_ref.get(),
      /*enable_logging=*/true, /*enable_inliner=*/false));

  std::string errorMessage;
  auto input_file = mlir::openInputFile(filepath, &errorMessage);
  EXPECT_THAT(input_file, Not(IsNull()));

  auto output_stream = mlir::openOutputFile(output_dump, &errorMessage);
  EXPECT_THAT(output_stream, Not(IsNull()));

  mlir::PassPipelineCLParser passPipeline(
      /*arg=*/"", /*description=*/"Compiler passes to run", /*alias=*/"p");
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::RegisterAllTensorFlowDialects(registry);

  mlir::registerAllPasses();
  mlir::registerTensorFlowPasses();

  EXPECT_TRUE(mlir::MlirOptMain(output_stream->os(), std::move(input_file),
                                registry,
                                mlir::MlirOptMainConfig{}
                                    .splitInputFile("")
                                    .verifyPasses(false)
                                    .allowUnregisteredDialects(false)
                                    .setPassPipelineParser(passPipeline))
                  .succeeded());
}

TEST(DumpRawStringToFileTest, Valid) {
  toolchain::StringRef example = "module {\n}";
  setenv("TF_DUMP_GRAPH_PREFIX", testing::TmpDir().c_str(), 1);

  std::string filepath = DumpRawStringToFile("example", example);
  ASSERT_NE(filepath, "(TF_DUMP_GRAPH_PREFIX not specified)");
  ASSERT_NE(filepath, "LOG(INFO)");
  ASSERT_NE(filepath, "(unavailable)");

  Env* env = Env::Default();
  std::string file_txt_module;
  TF_ASSERT_OK(ReadFileToString(env, filepath, &file_txt_module));
  EXPECT_EQ(file_txt_module, example);
}

}  // namespace
}  // namespace machina
