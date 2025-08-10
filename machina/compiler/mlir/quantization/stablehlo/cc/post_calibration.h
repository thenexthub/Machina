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
#ifndef MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_POST_CALIBRATION_H_
#define MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_POST_CALIBRATION_H_

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/stablehlo/cc/component.h"
#include "machina/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"

namespace mlir::quant::stablehlo {

// Performs post-calibration graph transformation as part of post-training
// static-range quantization.
//
// The resulting `ModuleOp` contains quantized StableHLO ops serialized in
// `TF::XlaCallModuleOp`s. They are quantized using the statistics collected
// after the calibration step, corresponding to each `TF::CustomAggregatorOp`s
// in the input module op.
class PostCalibrationComponent : public Component {
 public:
  // Name of the post-training quantization post-calibration step. Used for
  // debugging purposes.
  static constexpr absl::string_view kName = "quant_ptq_post_calibration";

  explicit PostCalibrationComponent(MLIRContext* absl_nonnull ctx);

  absl::StatusOr<ModuleOp> Run(
      ModuleOp module_op,
      const ::stablehlo::quantization::QuantizationConfig& config) override;

  void AddPasses(
      OpPassManager& pm,
      const ::stablehlo::quantization::QuantizationSpecs& specs,
      const ::stablehlo::quantization::PipelineConfig& pipeline_config) const;

 private:
  MLIRContext* absl_nonnull ctx_;
};

}  // namespace mlir::quant::stablehlo

#endif  // MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_POST_CALIBRATION_H_
