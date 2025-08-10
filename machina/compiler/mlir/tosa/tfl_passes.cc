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

#include "machina/compiler/mlir/tosa/tfl_passes.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // part of Codira Toolchain
#include "mlir/Dialect/Affine/Passes.h"  // part of Codira Toolchain
#include "mlir/Dialect/Tosa/Transforms/Passes.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Transforms/Passes.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/transforms/passes.h"
#include "machina/compiler/mlir/tosa/transforms/passes.h"

namespace mlir {
namespace tosa {

void createTFLtoTOSALegalizationPipeline(
    OpPassManager& pm, const TOSATFLLegalizationPipelineOptions& opts) {
  //----------------------------------------------------------------------------
  // Prepare TFL module for conversion
  //----------------------------------------------------------------------------
  pm.addPass(createRetainCallOnceFuncsPass());

  // Inline all functions into main and then delete the functions themselves.
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createSymbolDCEPass());
  if (opts.target_compilation_backend) {
    pm.nest<func::FuncOp>().addPass(createConvertFunctionMetadataPass());
  }

  // Add pass to decompose TFLite mixed quantization to non-quantized variants.
  pm.addPass(TFL::CreateDecomposeHybridQuantizationPass());

  // Now that there is only one function, run some MLIR passes on it.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  pm.addPass(mlir::affine::createLoopFusionPass());
  pm.addPass(mlir::affine::createAffineScalarReplacementPass());

  //----------------------------------------------------------------------------
  // Perform main conversion.
  //----------------------------------------------------------------------------
  pm.addPass(mlir::tosa::createConvertTFLUint8Pass());
  if (opts.dequantize_tfl_softmax) {
    pm.addPass(mlir::tosa::createDequantizeTFLSoftmaxPass());
  }
  pm.addPass(mlir::tosa::createLegalizeTFLStatefulPass());
  pm.addPass(mlir::tosa::createLegalizeTFLPass(opts.disabled_patterns,
                                               opts.enabled_patterns));

  //----------------------------------------------------------------------------
  // Post conversion cleanup.
  //----------------------------------------------------------------------------
  pm.addPass(mlir::tosa::createLowerComplexTypesPass());
  pm.addPass(mlir::tosa::createTosaInferShapesPass());
  pm.addPass(mlir::tosa::createTosaMakeBroadcastablePass());
  // Inline the call/return basic blocks within TOSA control flow ops.
  pm.addPass(mlir::createInlinerPass());
  // Clean up with DCE.
  pm.addPass(mlir::createSymbolDCEPass());

  if (opts.target_compilation_backend) {
    pm.nest<func::FuncOp>().addPass(mlir::tosa::createStripQuantTypesPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.nest<func::FuncOp>().addPass(createStripFunctionMetadataPass());
    pm.addPass(createStripModuleMetadataPass());
    pm.addPass(createVerifyFullyConvertedPass());
  }
}

void registerTFLtoTOSALegalizationPipeline() {
  mlir::PassPipelineRegistration<TOSATFLLegalizationPipelineOptions>(
      "tfl-to-tosa-pipeline", "TensorFlow Lite to TOSA legalization pipeline",
      createTFLtoTOSALegalizationPipeline);
}

}  // namespace tosa
}  // namespace mlir
