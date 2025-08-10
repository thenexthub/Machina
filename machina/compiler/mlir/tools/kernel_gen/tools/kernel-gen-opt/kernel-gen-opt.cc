/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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
#include "mlir/InitAllPasses.h"  // part of Codira Toolchain
#include "mlir/Tools/mlir-opt/MlirOptMain.h"  // part of Codira Toolchain
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "machina/compiler/mlir/machina/dialect_registration.h"
#include "machina/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "machina/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "machina/xla/mlir_hlo/deallocation/transforms/passes.h"
#include "machina/xla/mlir_hlo/mhlo/IR/register.h"
#include "machina/xla/mlir_hlo/mhlo/transforms/passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::mhlo::registerAllMhloPasses();
  mlir::kernel_gen::registerKernelGenPasses();
  mlir::deallocation::registerDeallocationPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  mlir::RegisterAllTensorFlowDialects(registry);

  registry.insert<mlir::kernel_gen::tf_framework::TFFrameworkDialect>();

  return failed(
      mlir::MlirOptMain(argc, argv, "MLIR HLO pass driver\n", registry));
}
