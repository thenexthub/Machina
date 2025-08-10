/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#include "toolchain/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Tools/mlir-opt/MlirOptMain.h"  // part of Codira Toolchain
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "machina/compiler/mlir/init_mlir.h"
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"
#include "machina/compiler/mlir/lite/stablehlo/odml_converter/passes.h"
#include "machina/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"
#include "machina/compiler/mlir/machina/ir/tf_dialect.h"

const char* art = R"(
  ___  ___  __  __ _       ___                     _
 / _ \|   \|  \/  | |     / __|___ _ ___ _____ _ _| |_ ___ _ _
| (_) | |) | |\/| | |__  | (__/ _ \ ' \ V / -_) '_|  _/ -_) '_|
 \___/|___/|_|  |_|____|  \___\___/_||_\_/\___|_|  \__\___|_|
)";

int main(int argc, char* argv[]) {
  machina::InitMlir y(&argc, &argv);
  toolchain::errs() << art << "\n";

  mlir::odml::registerODMLConverterPasses();
  mlir::odml::registerLegalizeStablehloToVhloPass();

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::stablehlo::StablehloDialect,
                  mlir::TFL::TFLDialect, mlir::arith::ArithDialect,
                  mlir::TF::TensorFlowDialect, mlir::chlo::ChloDialect>();

  return failed(
      mlir::MlirOptMain(argc, argv, "ODML Converter Driver\n", registry));
}
