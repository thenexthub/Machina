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

#include <utility>

#include "absl/log/log.h"
#include "toolchain/Support/LogicalResult.h"
#include "mlir/IR/AsmState.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/DialectRegistry.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"  // part of Codira Toolchain
#include "mlir/Tools/mlir-translate/Translation.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/init_mlir.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/graph_debug_info.pb.h"
#include "machina/core/ir/dialect.h"
#include "machina/core/ir/importexport/graphdef_export.h"
#include "machina/core/ir/importexport/graphdef_import.h"
#include "machina/core/ir/importexport/load_proto.h"
#include "machina/core/platform/status.h"

namespace mlir {

TranslateToMLIRRegistration graphdef_to_mlir(
    "graphdef-to-mlir", "graphdef-to-mlir",
    [](StringRef proto_txt, MLIRContext *context) {
      machina::GraphDebugInfo debug_info;
      machina::GraphDef graphdef;
      absl::Status status = machina::LoadProtoFromBuffer(
          {proto_txt.data(), proto_txt.size()}, &graphdef);
      if (!status.ok()) {
        LOG(ERROR) << status.message();
        return OwningOpRef<mlir::ModuleOp>{};
      }
      auto errorOrModule = tfg::ImportGraphDef(context, debug_info, graphdef);
      if (!errorOrModule.ok()) {
        LOG(ERROR) << errorOrModule.status();
        return OwningOpRef<mlir::ModuleOp>{};
      }
      return std::move(errorOrModule.value());
    });

TranslateFromMLIRRegistration mlir_to_graphdef(
    "mlir-to-graphdef", "mlir-to-graphdef",
    [](ModuleOp module, raw_ostream &output) {
      machina::GraphDef graphdef;
      absl::Status status = mlir::tfg::ConvertToGraphDef(module, &graphdef);
      if (!status.ok()) {
        LOG(ERROR) << "Error exporting MLIR module to GraphDef: " << status;
        return failure();
      }
      output << graphdef.DebugString();
      return success();
    },
    [](DialectRegistry &registry) { registry.insert<tfg::TFGraphDialect>(); });
}  //  namespace mlir

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();
  machina::InitMlir y(&argc, &argv);
  return failed(
      mlir::mlirTranslateMain(argc, argv, "Graph(Def)<->TFG Translation Tool"));
}
