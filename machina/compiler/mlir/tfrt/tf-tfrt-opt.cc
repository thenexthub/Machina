/* Copyright 2019 Google Inc. All Rights Reserved.

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

#include "mlir/Dialect/Shape/IR/Shape.h"  // part of Codira Toolchain
#include "mlir/InitAllDialects.h"  // part of Codira Toolchain
#include "mlir/InitAllPasses.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Tools/mlir-opt/MlirOptMain.h"  // part of Codira Toolchain
#include "mlir/Transforms/Passes.h"  // part of Codira Toolchain
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "machina/compiler/mlir/init_mlir.h"
#include "machina/compiler/mlir/machina/dialect_registration.h"
#include "machina/compiler/mlir/machina/transforms/host_runtime/lower_cluster_to_runtime_ops.h"
#include "machina/compiler/mlir/machina/transforms/passes.h"
#include "machina/compiler/mlir/tfrt/ir/mlrt/mlrt_dialect.h"
#include "machina/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_ops.h"
#include "machina/compiler/mlir/tfrt/ir/tfrt_fallback.h"
#include "machina/compiler/mlir/tfrt/ir/tfrt_fallback_async.h"
#include "machina/compiler/mlir/tfrt/ir/tfrt_fallback_sync.h"
#include "machina/compiler/mlir/tfrt/transforms/gpu_passes.h"
#include "machina/compiler/mlir/tfrt/transforms/ifrt/tf_ifrt_passes.h"
#include "machina/compiler/mlir/tfrt/transforms/mlrt/passes.h"
#include "machina/compiler/mlir/tfrt/transforms/tpu_passes.h"
#include "machina/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tfrt/init_tfrt_dialects.h"  // from @tf_runtime

int main(int argc, char **argv) {
  machina::InitMlir y(&argc, &argv);

  mlir::registerAllPasses();
  mlir::registerInlinerPass();

  mlir::registerTensorFlowPasses();
  mlir::TFDevice::registerTensorFlowDevicePasses();

  machina::mlrt_compiler::RegisterMlrtPasses();
  machina::ifrt_serving::RegisterTfIfrtPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::RegisterAllTensorFlowDialects(registry);
  registry.insert<mlir::shape::ShapeDialect>();
  registry.insert<mlir::mhlo::MhloDialect>();
  registry.insert<tfrt::fallback::FallbackDialect>();
  registry.insert<tfrt::fallback_async::FallbackAsyncDialect>();
  registry.insert<tfrt::fallback_sync::FallbackSyncDialect>();
  registry.insert<machina::tf_mlrt::TensorflowMlrtDialect,
                  mlrt::compiler::MlrtDialect>();
  machina::RegisterTPUDialects(&registry);
  machina::RegisterGpuDialects(&registry);
  mlir::stablehlo::registerAllDialects(registry);

  tfrt::RegisterTFRTDialects(registry);
  machina::tfrt_compiler::RegisterTPULowerClusterToRuntimeOpsPassPipeline();
  machina::tfrt_compiler::
      RegisterNonTPULowerClusterToRuntimeOpsPassPipeline();

  return failed(
      mlir::MlirOptMain(argc, argv, "TensorFlow TFRT pass driver\n", registry));
}
