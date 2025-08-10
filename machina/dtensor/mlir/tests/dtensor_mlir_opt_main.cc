/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#include "toolchain/Support/LogicalResult.h"
#include "mlir/InitAllDialects.h"  // part of Codira Toolchain
#include "mlir/InitAllExtensions.h"  // part of Codira Toolchain
#include "mlir/InitAllPasses.h"  // part of Codira Toolchain
#include "mlir/Tools/mlir-opt/MlirOptMain.h"  // part of Codira Toolchain
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "machina/compiler/mlir/init_mlir.h"
#include "machina/compiler/mlir/machina/dialect_registration.h"
#include "machina/dtensor/mlir/create_dtensor_mlir_passes.h"
#include "machina/dtensor/mlir/dtensor_dialect/ir/dialect.h"
#include "machina/dtensor/mlir/ir/tf_dtensor.h"

int main(int argc, char **argv) {
  machina::InitMlir y(&argc, &argv);

  mlir::registerAllPasses();
  machina::dtensor::registerDTensorPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::dtensor::DTensorDialect>();
  mlir::TF::RegisterDTensorTFOps();

  return failed(
      mlir::MlirOptMain(argc, argv, "DTensor pass driver\n", registry));
}
