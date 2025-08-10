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

#include "toolchain/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/Quant/IR/Quant.h"  // part of Codira Toolchain
#include "mlir/Dialect/SCF/IR/SCF.h"  // part of Codira Toolchain
#include "mlir/Dialect/Shape/IR/Shape.h"  // part of Codira Toolchain
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // part of Codira Toolchain
#include "mlir/InitAllPasses.h"  // part of Codira Toolchain
#include "mlir/Tools/mlir-opt/MlirOptMain.h"  // part of Codira Toolchain
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "stablehlo/transforms/Passes.h"  // from @stablehlo
#include "machina/compiler/mlir/init_mlir.h"
#include "machina/compiler/mlir/quantization/common/ir/QuantOps.h"
#include "machina/compiler/mlir/quantization/stablehlo/cc/pass_pipeline.h"
#include "machina/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h"
#include "machina/compiler/mlir/quantization/stablehlo/passes/passes.h"
#include "machina/compiler/mlir/quantization/stablehlo/passes/testing/passes.h"
#include "machina/compiler/mlir/machina/ir/tf_dialect.h"
#include "machina/compiler/mlir/machina/ir/tf_executor.h"
#include "machina/compiler/mlir/machina/ir/tf_saved_model.h"
#include "machina/compiler/mlir/machina/transforms/passes.h"
#include "machina/xla/mlir_hlo/mhlo/IR/register.h"
#include "machina/xla/mlir_hlo/mhlo/transforms/passes.h"
#include "machina/core/ir/types/dialect.h"

int main(int argc, char** argv) {
  machina::InitMlir y(&argc, &argv);

  mlir::registerAllPasses();
  mlir::registerTensorFlowPasses();
  mlir::quant::stablehlo::registerPasses();
  mlir::quant::stablehlo::registerBridgePasses();
  mlir::stablehlo::registerPasses();
  mlir::mhlo::registerAllMhloPasses();
  // These passes are only used for testing purposes.
  mlir::quant::stablehlo::testing::registerTestPasses();

  // Register StableHLO Quantizer pass pipelines.
  mlir::quant::stablehlo::RegisterPassPipelines();

  mlir::DialectRegistry registry;
  registry.insert<mlir::scf::SCFDialect, mlir::TF::TensorFlowDialect,
                  mlir::tf_saved_model::TensorFlowSavedModelDialect,
                  mlir::func::FuncDialect, mlir::shape::ShapeDialect,
                  mlir::arith::ArithDialect, mlir::tf_type::TFTypeDialect,
                  mlir::quant::QuantDialect, mlir::tensor::TensorDialect,
                  mlir::stablehlo::StablehloDialect,
                  mlir::tf_executor::TensorFlowExecutorDialect,
                  mlir::vhlo::VhloDialect, mlir::quant::ir::TFQuantDialect>();
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::func::registerAllExtensions(registry);
  return failed(
      mlir::MlirOptMain(argc, argv, "StableHLO quant Pass Driver\n", registry));
}
