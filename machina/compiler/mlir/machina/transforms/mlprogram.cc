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

#include "machina/compiler/mlir/machina/transforms/mlprogram.h"

#include "toolchain/ADT/STLFunctionalExtras.h"
#include "toolchain/ADT/Twine.h"
#include "mlir/Transforms/Passes.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/transforms/bridge.h"
#include "machina/compiler/mlir/machina/transforms/passes.h"
#include "machina/compiler/mlir/machina/transforms/tf_saved_model_passes.h"
#include "machina/compiler/mlir/tf2xla/internal/clustering_bridge_passes.h"
#include "machina/compiler/mlir/tf2xla/transforms/passes.h"
#include "machina/xla/mlir_hlo/mhlo/transforms/passes.h"

namespace machina {

void PopulateLowerToMlProgramAndHloPipeline(mlir::OpPassManager& pm) {
  machina::tf2xla::internal::AddNonReplicatedBridgeClusteringPipelinePasses(
      pm);

  // Remove unused global tensors, or make then immutable if possible.
  pm.addPass(mlir::tf_saved_model::CreateOptimizeGlobalTensorsPass());

  pm.addPass(
      mlir::tf_saved_model::CreateConvertSessionInitializerToFunctionPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TFDevice::CreateDecomposeResourceOpsPass());
  pm.addPass(mlir::TF::CreateNameAnonymousIteratorsPass());

  // This will add regions to IfOp/WhileOp (turning them into IfRegionOp
  // and WhileRegionOp), but be aware that those regions will still contain
  // calls.
  pm.addPass(mlir::TF::CreateTFFunctionalControlFlowToRegions());

  pm.addPass(mlir::tf_saved_model::CreateLowerVariableOpsToMlProgramPass());
  pm.addPass(mlir::tf_saved_model::CreateLowerGlobalsToMlProgramPass());
  pm.addPass(mlir::TF::CreateLocalizeVarHandlesPass());
  pm.addPass(mlir::tf_saved_model::CreateAddFunctionsForExportedNamesPass());
  pm.addPass(mlir::tf_saved_model::CreateStripSavedModuleMetadataPass());

  pm.addPass(mlir::TF::CreateRemoveUnusedArgumentsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TF::CreateRemoveUnusedWhileResultsPass());

  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSymbolDCEPass());

  pm.addPass(mlir::TF::CreateTFShapeInferencePass());

  toolchain::StringRef tf2xla_fallback_device_type = "MACHINA_MACHINA_XLA_CPU_JIT";
  pm.addPass(mlir::mhlo::createLegalizeTFPass(
      /*legalize_chlo=*/true, tf2xla_fallback_device_type,
      /*prefer_tf2xla=*/false));

  pm.addPass(mlir::TF::CreateStripTfAttributesPass());

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  pm.addPass(mlir::TF::CreateOrderByDialectPass());

  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
}

}  // namespace machina
