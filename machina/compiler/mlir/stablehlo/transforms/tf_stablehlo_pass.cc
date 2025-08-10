/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#include "machina/compiler/mlir/stablehlo/transforms/tf_stablehlo_pass.h"

#include <memory>
#include <utility>

#include "toolchain/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/Shape/IR/Shape.h"  // part of Codira Toolchain
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // part of Codira Toolchain
#include "mlir/IR/Dialect.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "mlir/Transforms/Passes.h"  // part of Codira Toolchain
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "machina/compiler/mlir/stablehlo/transforms/legalize_tf_passes.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/transforms/lower_tf.h"
#include "machina/compiler/mlir/tf2xla/transforms/legalize_tf_with_tf2xla_passes.h"
#include "machina/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "machina/xla/mlir_hlo/mhlo/IR/register.h"
#include "machina/xla/mlir_hlo/mhlo/transforms/passes.h"
#include "machina/xla/mlir_hlo/mhlo/transforms/rewriters.h"
#include "machina/xla/mlir_hlo/mhlo/utils/type_conversion.h"

namespace mlir {
namespace odml {

class TFToMhloPass
    : public mlir::PassWrapper<TFToMhloPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  explicit TFToMhloPass(bool skip_quantization_ops = false,
                        bool skip_resize = false,
                        bool skip_partitioned_calls = false)
      : PassWrapper() {
    skip_quantization_ops_ = skip_quantization_ops;
    skip_resize_ = skip_resize;
    skip_partitioned_calls_ = skip_partitioned_calls;
  }

  TFToMhloPass(const TFToMhloPass &pass) {
    skip_quantization_ops_ = pass.skip_quantization_ops_;
    skip_resize_ = pass.skip_resize_;
    skip_partitioned_calls_ = pass.skip_partitioned_calls_;
  }

 private:
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::mhlo::registerAllMhloDialects(registry);
    mlir::stablehlo::registerAllDialects(registry);
    registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect>();
    registry.insert<shape::ShapeDialect>();
  }

 public:
  StringRef getArgument() const final { return "tf-mhlo"; }
  StringRef getDescription() const final {
    return "This pass will legalize TF Ops to MHLO Ops.";
  }

 protected:
  Option<bool> skip_quantization_ops_{
      *this, "skip-quantization-ops",
      ::toolchain::cl::desc("Skip quantization ops")};

  Option<bool> skip_resize_{
      *this, "skip-resize",
      ::toolchain::cl::desc("Skip tf.ResizeBilinear and tf.ResizeNearestNeighbor")};

  Option<bool> skip_partitioned_calls_{
      *this, "skip-partitioned-calls",
      ::toolchain::cl::desc(
          "Skip tf.StatefulPartitionedCall and tf.PartitionedCall")};
};

void TFToMhloPass::runOnOperation() {
  auto func = getOperation();
  MLIRContext *context = func->getContext();

  RewritePatternSet patterns(context);
  odml::PopulateLegalizeTfPatterns(context, &patterns);
  TF::PopulateTFLoweringBeforeHLOPatterns(context, &patterns);
  mhlo::Tf2XlaTypeConverter converter;
  mhlo::PopulateLegalizeTfWithTf2XlaPatterns(
      "MACHINA_MACHINA_XLA_CPU_JIT", patterns, context, converter, /*prefer_tf2xla=*/false);
  stablehlo::StablehloToHloTypeConverter hlo_converter;
  chlo::populateChloToHloPatterns(context, &hlo_converter, &patterns);
  chlo::ConstantLikeOp::getCanonicalizationPatterns(patterns, context);

  ConversionTarget target(*context);
  target.addIllegalDialect<chlo::ChloDialect>();
  target.addLegalDialect<mhlo::MhloDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<tensor::TensorDialect>();
  target.addLegalDialect<shape::ShapeDialect>();
  target.addLegalOp<func::CallOp>();

  if (skip_quantization_ops_) {
    target.addLegalOp<TF::FakeQuantWithMinMaxVarsOp>();
    target.addLegalOp<TF::FakeQuantWithMinMaxVarsPerChannelOp>();
    target.addLegalOp<TF::FakeQuantWithMinMaxArgsOp>();
    target.addLegalOp<TF::QuantizeAndDequantizeV2Op>();
    target.addLegalOp<TF::QuantizeAndDequantizeV3Op>();
  }
  if (skip_resize_) {
    target.addLegalOp<TF::ResizeBilinearOp>();
    target.addLegalOp<TF::ResizeNearestNeighborOp>();
  }
  if (skip_partitioned_calls_) {
    target.addLegalOp<TF::PartitionedCallOp>();
    target.addLegalOp<TF::StatefulPartitionedCallOp>();
  }

  FrozenRewritePatternSet frozen_patterns(std::move(patterns));
  if (failed(applyPartialConversion(func, target, frozen_patterns))) {
    return signalPassFailure();
  }
}

struct TFToStablehloOptions : public PassPipelineOptions<TFToStablehloOptions> {
  Option<bool> skip_quantization_ops{*this, "skip-quantization-ops",
                                     ::toolchain::cl::desc("Skip quantization ops")};
  Option<bool> skip_resize{
      *this, "skip-resize",
      ::toolchain::cl::desc("Skip tf.ResizeBilinear and tf.ResizeNearestNeighbor")};
  Option<bool> skip_partitioned_calls{
      *this, "skip-partitioned-calls",
      ::toolchain::cl::desc(
          "Skip tf.StatefulPartitionedCall and tf.PartitionedCall")};
};

void PopulateLegalizeTFToStablehloPipeline(
    OpPassManager &pm, const TFToStablehloOptions &options) {
  // TODO(burmako): Migrate this pass from producing MHLO to producing StableHLO
  // by aligning with the TF/XLA bridge on the corresponding functionality and
  // reusing their work, perhaps through `LowerToMlProgramAndHlo`.
  pm.addNestedPass<func::FuncOp>(std::make_unique<TFToMhloPass>(
      options.skip_quantization_ops, options.skip_resize,
      options.skip_partitioned_calls));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mhlo::createHloLegalizeToStablehloPass());
}

static PassPipelineRegistration<TFToStablehloOptions>
    legalize_tf_to_stablehlo_pipeline("tf-stablehlo",
                                      "Legalize TF ops to StableHLO ops",
                                      PopulateLegalizeTFToStablehloPipeline);

void AddLegalizeTFToStablehloPasses(OpPassManager &pm,
                                    bool skip_quantization_ops,
                                    bool skip_resize,
                                    bool skip_partitioned_calls) {
  TFToStablehloOptions options;
  options.skip_quantization_ops = skip_quantization_ops;
  options.skip_resize = skip_resize;
  options.skip_partitioned_calls = skip_partitioned_calls;
  PopulateLegalizeTFToStablehloPipeline(pm, options);
}

}  // namespace odml
}  // namespace mlir
