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
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain  // IWYU pragma: keep
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/Quant/IR/Quant.h"  // part of Codira Toolchain  // IWYU pragma: keep
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain  // IWYU pragma: keep
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "machina/compiler/mlir/quantization/common/ir/QuantOps.h"  // IWYU pragma: keep
#include "machina/compiler/mlir/quantization/stablehlo/passes/passes.h"
#include "machina/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "machina/compiler/mlir/quantization/machina/cc/run_passes.h"
#include "machina/compiler/mlir/quantization/machina/quantization_options.pb.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"  // IWYU pragma: keep

#define DEBUG_TYPE "quantize-composite-functions"

namespace mlir::quant::stablehlo {

#define GEN_PASS_DEF_QUANTIZECOMPOSITEFUNCTIONSPASS
#include "machina/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

namespace {

using ::machina::quantization::RunPassesOnModuleOp;

class QuantizeCompositeFunctionsPass
    : public impl::QuantizeCompositeFunctionsPassBase<
          QuantizeCompositeFunctionsPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantizeCompositeFunctionsPass)

  using impl::QuantizeCompositeFunctionsPassBase<
      QuantizeCompositeFunctionsPass>::QuantizeCompositeFunctionsPassBase;

  explicit QuantizeCompositeFunctionsPass(
      const bool enable_per_channel_quantized_weight) {
    enable_per_channel_quantized_weight_ = enable_per_channel_quantized_weight;
  }

 private:
  void runOnOperation() override;
};

void QuantizeCompositeFunctionsPass::runOnOperation() {
  MLIRContext& ctx = getContext();

  PassManager pm(&ctx);
  // Intermediate output from QuantizePass will have quantized ops
  // (XlaCallModuleOps) with quantized input and output types, which are not
  // allowed in the TF dialect.
  pm.enableVerifier(false);

  PrepareQuantizePassOptions options;
  options.enable_per_channel_quantized_weight_ =
      enable_per_channel_quantized_weight_;
  // Change this to user-given bit width once we have custom configuration.
  options.bit_width_ = 8;

  // Insert quantization parameters for weights for ops with `weight_only_ptq`
  // attribute.
  pm.addNestedPass<func::FuncOp>(createInsertWeightParamPass());

  // PrepareQuantizePass uses SymbolTable to fetch relevant GEMM ops for
  // determining quantization attributes. This requires module-level context.
  pm.addPass(createPrepareQuantizePass(options));

  QuantizePassOptions quantize_options;
  quantize_options.enable_per_channel_quantized_weight_ =
      enable_per_channel_quantized_weight_;

  // QuantizePass modifies FuncOps referenced outside of its given scope
  // and therefore requires a module-level context.
  pm.addPass(createQuantizePass(quantize_options));
  pm.addNestedPass<func::FuncOp>(createPostQuantizePass());

  // Convert XlaCallModuleOps lifted but not quantized to func.call op.
  // The reasons these ops are not quantized may be:
  // 1. Disabled due to selective quantization.
  // 2. Not supported, e.g. add op for server.
  pm.addPass(createXlaCallModuleToCallPass());

  // TODO: b/321729008 - move this implementation to quantization_patterns.cc.
  if (merge_fusion_with_dequantize_) {
    pm.addPass(createMergeFusionWithDequantizePass());
  }

  ModuleOp module_op = getOperation();
  if (const absl::Status pm_run_status =
          RunPassesOnModuleOp(mlir_dump_file_name_, pm, module_op);
      !pm_run_status.ok()) {
    signalPassFailure();
  }
}

}  // namespace
}  // namespace mlir::quant::stablehlo
