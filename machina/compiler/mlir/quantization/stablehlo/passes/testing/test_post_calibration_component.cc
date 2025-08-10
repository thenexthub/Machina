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
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain  // IWYU pragma: keep
#include "mlir/Dialect/Quant/IR/Quant.h"  // part of Codira Toolchain  // IWYU pragma: keep
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain  // IWYU pragma: keep
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "machina/compiler/mlir/quantization/common/ir/QuantOps.h"  // IWYU pragma: keep
#include "machina/compiler/mlir/quantization/stablehlo/cc/config.h"
#include "machina/compiler/mlir/quantization/stablehlo/cc/post_calibration.h"
#include "machina/compiler/mlir/quantization/stablehlo/passes/testing/passes.h"  // IWYU pragma: keep
#include "machina/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "machina/compiler/mlir/machina/ir/tf_dialect.h"  // IWYU pragma: keep
#include "machina/compiler/mlir/machina/ir/tf_executor.h"  // IWYU pragma: keep
#include "machina/xla/mlir_hlo/mhlo/IR/hlo_ops.h"  // IWYU pragma: keep

namespace mlir::quant::stablehlo::testing {

#define GEN_PASS_DEF_TESTPOSTCALIBRATIONCOMPONENTPASS
#include "machina/compiler/mlir/quantization/stablehlo/passes/testing/passes.h.inc"

namespace {

using ::stablehlo::quantization::ExpandPresets;
using ::stablehlo::quantization::PipelineConfig;
using ::stablehlo::quantization::QuantizationConfig;

class TestPostCalibrationComponentPass
    : public impl::TestPostCalibrationComponentPassBase<
          TestPostCalibrationComponentPass> {
 public:
  using impl::TestPostCalibrationComponentPassBase<
      TestPostCalibrationComponentPass>::TestPostCalibrationComponentPassBase;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPostCalibrationComponentPass)

 private:
  void runOnOperation() override;
};

void TestPostCalibrationComponentPass::runOnOperation() {
  ModuleOp module_op = getOperation();
  MLIRContext& ctx = getContext();

  OpPassManager pm(ModuleOp::getOperationName());

  QuantizationConfig config = QuantizationConfig::default_instance();
  config.mutable_static_range_ptq_preset();

  const QuantizationConfig new_config = ExpandPresets(config);

  PipelineConfig pipeline_config;
  pipeline_config.set_unpack_quantized_types(unpack_quantized_types_);

  quant::stablehlo::PostCalibrationComponent component(&ctx);
  component.AddPasses(pm, new_config.specs(), pipeline_config);

  if (failed(runPipeline(pm, module_op))) {
    signalPassFailure();
  }
}

}  // namespace

}  // namespace mlir::quant::stablehlo::testing
