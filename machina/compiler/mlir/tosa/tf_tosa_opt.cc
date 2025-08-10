/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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

#include "mlir/InitAllPasses.h"               // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"       // part of Codira Toolchain
#include "mlir/Tools/mlir-opt/MlirOptMain.h"  // part of Codira Toolchain
#include "mlir/Transforms/Passes.h"           // part of Codira Toolchain
#include "machina//compiler/mlir/machina/transforms/tf_saved_model_passes.h"
#include "machina/compiler/mlir/init_mlir.h"
#include "machina/compiler/mlir/lite/transforms/passes.h"
#include "machina/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h"
#include "machina/compiler/mlir/register_common_dialects.h"
#include "machina/compiler/mlir/machina/transforms/host_runtime/lower_cluster_to_runtime_ops.h"
#include "machina/compiler/mlir/machina/transforms/host_runtime/runtime_passes.h"
#include "machina/compiler/mlir/machina/transforms/passes.h"
#include "machina/compiler/mlir/machina/transforms/sparsecore/sparsecore_passes.h"
#include "machina/compiler/mlir/machina/transforms/test_passes.h"
#include "machina/compiler/mlir/machina/transforms/tf_graph_optimization_pass.h"
#include "machina/compiler/mlir/machina/utils/mlprogram_util.h"
#include "machina/compiler/mlir/tf2xla/api/v1/compile_mlir_util.h"
#include "machina/compiler/mlir/tf2xla/internal/passes/clustering_passes.h"
#include "machina/compiler/mlir/tf2xla/internal/passes/mlir_to_graph_passes.h"
#include "machina/compiler/mlir/tf2xla/transforms/passes.h"
#include "machina/compiler/mlir/tosa/tf_passes.h"
#include "machina/compiler/mlir/tosa/tf_tfl_passes.h"
#include "machina/compiler/mlir/tosa/tfl_passes.h"
#include "machina/compiler/mlir/tosa/transforms/passes.h"
#include "machina/xla/mlir/framework/transforms/passes.h"
#include "machina/xla/mlir_hlo/mhlo/transforms/passes.h"

int main(int argc, char** argv) {
  machina::InitMlir y(&argc, &argv);

  mlir::registerAllPasses();
  mlir::registerTransformsPasses();
  mlir::registerTensorFlowPasses();
  mlir::TFDevice::registerTensorFlowDevicePasses();
  mlir::tf_saved_model::registerTensorFlowSavedModelPasses();
  mlir::TFL::registerTensorFlowLitePasses();
  mlir::mhlo::registerAllMhloPasses();

  // These are in compiler/mlir/tf2xla and not part of the above MHLO passes.
  mlir::mhlo::registerLegalizeTfPasses();
  mlir::mhlo::registerTfXlaPasses();
  mlir::quant::stablehlo::registerBridgePasses();
  machina::tf2xla::internal::registerTFXLABridgeClusteringPasses();
  machina::tf2xla::internal::registerTFXLABridgeMlirToGraphPasses();
  mlir::tf_test::registerTensorFlowTestPasses();
  mlir::xla_framework::registerXlaFrameworkPasses();
  machina::RegisterConvertMlirToXlaHloPipelineWithDefaults();
  machina::RegisterGraphOptimizationPasses();
  machina::RegisterMlProgramPasses();
  mlir::TFTPU::registerRuntimeLoweringPasses();
  mlir::TFDevice::registerSparseCorePasses();
  mlir::tosa::registerLegalizeTosaPasses();
  mlir::tosa::registerTFtoTOSALegalizationPipeline();
  mlir::tosa::registerTFLtoTOSALegalizationPipeline();
  mlir::tosa::registerTFTFLtoTOSALegalizationPipeline();

  machina::tfrt_compiler::RegisterTPULowerClusterToRuntimeOpsPassPipeline();
  machina::tfrt_compiler::
      RegisterNonTPULowerClusterToRuntimeOpsPassPipeline();

  mlir::DialectRegistry registry;
  mlir::RegisterCommonToolingDialects(registry);
  registry.insert<mlir::TFL::TensorFlowLiteDialect>();
  registry.insert<mlir::quantfork::QuantizationForkDialect>();

  return failed(
      mlir::MlirOptMain(argc, argv, "TensorFlow pass driver\n", registry));
}
