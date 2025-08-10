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
#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain  // IWYU pragma: keep
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain  // IWYU pragma: keep
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "machina/compiler/mlir/quantization/stablehlo/cc/config.h"
#include "machina/compiler/mlir/quantization/stablehlo/cc/pre_calibration.h"
#include "machina/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "machina/compiler/mlir/quantization/machina/quantization_options.pb.h"
#include "machina/compiler/mlir/machina/ir/tf_dialect.h"  // IWYU pragma: keep
#include "machina/compiler/mlir/machina/ir/tf_executor.h"  // IWYU pragma: keep
#include "machina/xla/mlir_hlo/mhlo/IR/hlo_ops.h"  // IWYU pragma: keep

namespace mlir::quant::stablehlo::testing {

#define GEN_PASS_DEF_TESTPRECALIBRATIONCOMPONENTPASS
#include "machina/compiler/mlir/quantization/stablehlo/passes/testing/passes.h.inc"

namespace {

using ::stablehlo::quantization::ExpandPresets;
using ::stablehlo::quantization::PopulateDefaults;
using ::stablehlo::quantization::QuantizationConfig;

class TestPreCalibrationComponentPass
    : public impl::TestPreCalibrationComponentPassBase<
          TestPreCalibrationComponentPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPreCalibrationComponentPass)

 private:
  void runOnOperation() override;
};

void TestPreCalibrationComponentPass::runOnOperation() {
  ModuleOp module_op = getOperation();
  MLIRContext& ctx = getContext();

  // Simply runs the PreCalibrationComponent with a default configuration.
  PreCalibrationComponent component(&ctx);
  QuantizationConfig quantization_config{};
  quantization_config.mutable_static_range_ptq_preset();
  quantization_config = ExpandPresets(PopulateDefaults(quantization_config));
  if (!component.Run(module_op, quantization_config).ok()) {
    signalPassFailure();
  }
}

}  // namespace
}  // namespace mlir::quant::stablehlo::testing
