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
// This transformation pass applies quantization propagation on TF dialect.

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "toolchain/Support/CommandLine.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/Quant/IR/Quant.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/DialectRegistry.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Matchers.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/IR/SymbolTable.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/IR/ValueRange.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassRegistry.h"  // part of Codira Toolchain
#include "mlir/Rewrite/FrozenRewritePatternSet.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/common/ir/QuantOps.h"
#include "machina/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "machina/compiler/mlir/quantization/machina/ops/tf_op_quant_spec.h"
#include "machina/compiler/mlir/quantization/machina/passes/passes.h"
#include "machina/compiler/mlir/machina/ir/tf_dialect.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

//===----------------------------------------------------------------------===//
// The preprocess-op Pass.
//
namespace mlir {
namespace quant {

namespace {

using QuantMethod =
    ::machina::quantization::QuantizationMethod::PresetMethod;
using QuantizationUnit = std::pair<Operation*, int>;
using QuantizationUnits = toolchain::SetVector<QuantizationUnit>;
using ::machina::quantization::OpSet;

// Preprocesses ops to allow multi-axis quantization, prior to quantization
// passes. Currently, per-channel quantization only supports 1D results.
class PreprocessOpPass
    : public PassWrapper<PreprocessOpPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect, QuantDialect,
                    mlir::quant::ir::TFQuantDialect>();
  }

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PreprocessOpPass)

  explicit PreprocessOpPass() = default;

  // Constructor used by manually creating the pass.
  explicit PreprocessOpPass(OpSet op_set, const QuantMethod quantization_method,
                            bool enable_per_channel_quantization) {
    op_set_ = op_set;
    quantization_method_ = quantization_method;
    enable_per_channel_quantization_ = enable_per_channel_quantization;
  }

  PreprocessOpPass(const PreprocessOpPass& other) {
    op_set_ = other.op_set_;
    quantization_method_ = other.quantization_method_;
    enable_per_channel_quantization_ = other.enable_per_channel_quantization_;
  }

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-preprocess-op";
  }
  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Preprocess TF op prior to quantization";
  }

  void runOnOperation() override;

 private:
  Option<OpSet> op_set_{
      *this, "target-opset", toolchain::cl::init(OpSet::UNIFORM_QUANTIZED),
      toolchain::cl::desc("Choose target opset."),
      toolchain::cl::values(
          clEnumValN(OpSet::TF, "TF",
                     "Uses TF ops that mimic quantization behavior"),
          clEnumValN(OpSet::XLA, "XLA", "Uses TF XLA ops"),
          clEnumValN(OpSet::UNIFORM_QUANTIZED, "UNIFORM_QUANTIZED",
                     "Uses TF Uniform Quantized ops"))};

  Option<QuantMethod> quantization_method_{
      *this, "quantization-method",
      toolchain::cl::init(machina::quantization::QuantizationMethod::
                         METHOD_STATIC_RANGE_INT8),
      toolchain::cl::desc("Choose quantization method."),
      toolchain::cl::values(
          clEnumValN(machina::quantization::QuantizationMethod::
                         METHOD_STATIC_RANGE_INT8,
                     "ptq", "Post-training static-range quantization"),
          clEnumValN(machina::quantization::QuantizationMethod::
                         METHOD_DYNAMIC_RANGE_INT8,
                     "drq", "Post-training dynamic-range quantizaiton"),
          clEnumValN(machina::quantization::QuantizationMethod::
                         METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8,
                     "weight_only", "Post-training weight-only quantizaiton"))};

  Option<bool> enable_per_channel_quantization_{
      *this, "enable-per-channel-quantization", toolchain::cl::init(false),
      toolchain::cl::desc("Whether enable per-channel quantized weights.")};
};

// Apply constant transformations for the op_set.
class PreprocessConstantOp : public OpRewritePattern<TF::PartitionedCallOp> {
 public:
  explicit PreprocessConstantOp(MLIRContext* context, OpSet op_set,
                                QuantMethod quantization_method,
                                bool enable_per_channel_quantization)
      : OpRewritePattern<TF::PartitionedCallOp>(context),
        op_set_(op_set),
        quantization_method_(quantization_method),
        enable_per_channel_quantization_(enable_per_channel_quantization) {}

  LogicalResult addReshapeOpToDepthwiseWeight(TF::PartitionedCallOp op,
                                              PatternRewriter& rewriter,
                                              StringRef function_name) const {
    std::unique_ptr<OpQuantSpec> spec = GetTFOpQuantSpec(op);
    const absl::flat_hash_set<int> operands = spec->quantizable_operands;

    if (operands.size() != 1) return failure();
    int weight_operand_idx = *operands.begin();

    Operation* weight_op = op.getOperand(weight_operand_idx).getDefiningOp();
    DenseFPElementsAttr attr;
    if (!matchPattern(weight_op->getResult(0), m_Constant(&attr))) {
      return failure();
    }

    // Get new shape.
    toolchain::ArrayRef<int64_t> cur_shape = attr.getType().getShape();
    int cur_rank = cur_shape.size();
    if (cur_rank != 4 || cur_shape[2] == 1) return failure();
    TensorType new_shape = RankedTensorType::get(
        {cur_shape[0], cur_shape[1], 1, cur_shape[2] * cur_shape[3]},
        attr.getElementType());

    // Inserts a reshape op.
    auto shape_spec_type =
        RankedTensorType::get({cur_rank}, rewriter.getIntegerType(64));
    auto new_shape_const_attr =
        DenseElementsAttr::get(shape_spec_type, new_shape.getShape());
    rewriter.setInsertionPointAfter(weight_op);
    auto new_shape_const = rewriter.create<arith::ConstantOp>(
        weight_op->getLoc(), shape_spec_type, new_shape_const_attr);
    auto reshape_op = rewriter.create<TF::ReshapeOp>(
        weight_op->getLoc(), new_shape, weight_op->getResult(0),
        new_shape_const);
    op->setOperand(weight_operand_idx, reshape_op);

    // Create a new function with preprocessed types.
    ModuleOp module = op->getParentOfType<ModuleOp>();
    SymbolTable symbol_table(module);
    func::FuncOp float_func =
        dyn_cast<func::FuncOp>(symbol_table.lookup(function_name));
    OperandRange func_args = op.getArgs();
    func::FuncOp new_float_func = float_func.clone();

    SmallVector<Value> new_float_func_args{func_args.begin(), func_args.end()};
    new_float_func_args[weight_operand_idx] = reshape_op;
    new_float_func.getArgument(weight_operand_idx).setType(new_shape);
    new_float_func.setType(FunctionType::get(
        getContext(), TypeRange{ValueRange{new_float_func_args}},
        new_float_func.getResultTypes()));
    symbol_table.insert(new_float_func);

    op->setAttr("f", SymbolRefAttr::get(rewriter.getContext(),
                                        new_float_func.getName()));

    return success();
  }

  LogicalResult matchAndRewrite(TF::PartitionedCallOp op,
                                PatternRewriter& rewriter) const override {
    const auto f_attr = mlir::dyn_cast<FlatSymbolRefAttr>(op.getFAttr());
    // Non-quantizable op
    if (!op->hasAttr(kQuantTraitAttrName)) return failure();
    StringRef function_name = f_attr.getValue();
    // TODO(b/228928859): Improve the getter function to match attributes rather
    // than function name.
    if (!function_name.starts_with("composite_")) {
      return failure();
    }

    if (function_name.contains("depthwise_conv2d")) {
      // Uniform Quantized op requires weights of tf.DepthwiseConv2dNative to
      // be transformed from [H,W,C,M] to [H,W,1,CxM] where
      // H=height,W=width,C=channel,M=multiplier. Therefore, a reshape op is
      // inserted between the constant op and the function op so that the
      // constant is safely transformed for the multi-use cases as well. Note
      // that bias doesn't need transformation as its shape is already in [CxM].
      if (op_set_ == OpSet::UNIFORM_QUANTIZED ||
          (op_set_ == OpSet::XLA && enable_per_channel_quantization_ &&
           quantization_method_ ==
               machina::quantization::QuantizationMethod::
                   METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8)) {
        return addReshapeOpToDepthwiseWeight(op, rewriter, function_name);
      }
    }
    return failure();
  }

 private:
  const OpSet op_set_;
  const QuantMethod quantization_method_;
  const bool enable_per_channel_quantization_;
};

#include "machina/compiler/mlir/quantization/machina/passes/preprocess_op.inc"

void PreprocessOpPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  ModuleOp module_op = getOperation();

  populateWithGenerated(patterns);
  patterns.add<PreprocessConstantOp>(ctx, op_set_, quantization_method_,
                                     enable_per_channel_quantization_);
  FrozenRewritePatternSet frozen_patterns(std::move(patterns));

  for (auto func : module_op.getOps<func::FuncOp>()) {
    if (failed(applyPatternsGreedily(func, frozen_patterns))) {
      func.emitError() << "quant-preprocess-op failed.";
      signalPassFailure();
    }
  }
}

}  // namespace

// Creates an instance of the TensorFlow dialect PreprocessOp
// pass.
std::unique_ptr<OperationPass<ModuleOp>> CreatePreprocessOpPass(
    const OpSet op_set, QuantMethod quantization_method,
    const bool enable_per_channel_quantization) {
  return std::make_unique<PreprocessOpPass>(op_set, quantization_method,
                                            enable_per_channel_quantization);
}

static PassRegistration<PreprocessOpPass> pass;

}  // namespace quant
}  // namespace mlir
