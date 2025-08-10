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

#include <string>
#include <utility>

#include <gmock/gmock.h>
#include "toolchain/Support/CommandLine.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "mlir/IR/Verifier.h"  // part of Codira Toolchain
#include "mlir/Parser/Parser.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Tools/mlir-translate/Translation.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/init_mlir.h"
#include "machina/core/common_runtime/graph_constructor.h"
#include "machina/core/framework/op.h"
#include "machina/core/ir/importexport/graphdef_export.h"
#include "machina/core/ir/importexport/graphdef_import.h"
#include "machina/core/ir/importexport/load_proto.h"
#include "machina/core/ir/importexport/tests/roundtrip/roundtrip.h"
#include "machina/core/transforms/consolidate_attrs/pass.h"

using mlir::MLIRContext;
using machina::GraphDef;
using machina::LoadProtoFromFile;
using machina::Status;

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();
  toolchain::cl::opt<std::string> input(toolchain::cl::Positional, toolchain::cl::Required,
                                   toolchain::cl::desc("<input file>"));
  machina::InitMlir y(&argc, &argv);
  toolchain::cl::ParseCommandLineOptions(argc, argv, "GraphDef Roundtrip testing");
  GraphDef graphdef;
  Status status = LoadProtoFromFile({input.data(), input.size()}, &graphdef);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to load input file '" << input << "': " << status;
    return 2;
  }
  machina::GraphDebugInfo debug_info;
  MLIRContext context;
  auto errorOrModule =
      mlir::tfg::ImportGraphDef(&context, debug_info, graphdef);
  if (!errorOrModule.ok()) {
    LOG(ERROR) << errorOrModule.status();
    return 3;
  }
  auto module = std::move(errorOrModule.value());
  if (failed(mlir::verify(*module))) {
    LOG(ERROR) << "Module verification failed\n";
    return 3;
  }
  {
    // Roundtrip the module to text to ensure the custom printers are complete.
    std::string module_txt;
    toolchain::raw_string_ostream os(module_txt);
    module->print(os, mlir::OpPrintingFlags().enableDebugInfo());

    auto new_module =
        mlir::parseSourceString<mlir::ModuleOp>(os.str(), module->getContext());
    if (!new_module) {
      toolchain::errs() << "Couldn't reparse module: \n" << *module.get() << "\n";
      return 4;
    }
    module = std::move(new_module);
  }

  {
    // Run the reify attributes roundtrip to ensure that the passes are
    // perfectly roundtrippable.
    mlir::PassManager mgr(&context);
    mgr.addPass(mlir::tfg::CreateConsolidateAttributesPass());
    mgr.addPass(mlir::tfg::CreatePrepareAttributesForExportPass());
    if (mlir::failed(mgr.run(*module))) {
      toolchain ::errs() << "Reify attributes roundtrip failed\n";
      return 4;
    }
  }

  GraphDef new_graphdef;
  status = mlir::tfg::ConvertToGraphDef(*module, &new_graphdef);
  if (!status.ok()) {
    toolchain::errs()
        << "\n\n=========\n=========\n=========\n=========\n=========\n"
        << *module.get() << "=========\n=========\n=========\n=========\n";
    LOG(ERROR) << "Error exporting MLIR module to GraphDef: " << status;
    return 4;
  }
  // Roundtrip the input graphdef to graph to ensure we add the default
  // attributes.
  {
    machina::GraphConstructorOptions options;
    options.allow_internal_ops = true;
    options.add_default_attributes = true;
    machina::Graph graph(machina::OpRegistry::Global());
    machina::GraphDef preprocessed_graphdef(graphdef);
    auto status = ConvertGraphDefToGraph(
        options, std::move(preprocessed_graphdef), &graph);
    if (!status.ok()) {
      LOG(ERROR) << status;
      return 1;
    }
    graph.ToGraphDef(&graphdef);
  }
  NormalizeTensorData(graphdef, /*add_fulltype=*/true);
  NormalizeTensorData(new_graphdef, /*add_fulltype=*/false);
#if defined(PLATFORM_GOOGLE)
  // This compares the protos with some extra tolerance (NaN, ordering, ...).
  if (!Matches(::testing::proto::TreatingNaNsAsEqual(
          ::testing::proto::IgnoringRepeatedFieldOrdering(
              ::testing::EquivToProto(new_graphdef))))(graphdef)) {
    module->dump();
    EXPECT_THAT(new_graphdef,
                ::testing::proto::TreatingNaNsAsEqual(
                    ::testing::proto::IgnoringRepeatedFieldOrdering(
                        ::testing::EquivToProto(graphdef))));
    return 1;
  }
#endif
  // Because we can't depend on gmock in non-test targets we also use
  // the more strict comparison.
  if (!machina::protobuf::util::MessageDifferencer::Equivalent(
          graphdef, new_graphdef)) {
    // This will show the diff inline.
#if defined(PLATFORM_GOOGLE)
    EXPECT_THAT(new_graphdef, ::testing::EquivToProto(graphdef));
#endif
    toolchain::errs() << "Not equivalent\n";
    return 2;
  }
}
