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

// The TF dialect uses some TF types that are illegal in the MHLO dialect and
// some generic types that are legal in MHLO. This pass legalizes TF types into
// types that are legal in MHLO. For example, TF::Qint8Type is converted to i8.
// Rewrites here should run before TF to MHLO op legalizations are run.

#include <memory>
#include <string>
#include <utility>

#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypeInterfaces.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Matchers.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/IR/TypeUtilities.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/stablehlo/utils/tf_type_utils.h"
#include "machina/compiler/mlir/machina/ir/tf_attributes.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/core/lib/monitoring/counter.h"

namespace mlir::quant::stablehlo {
namespace {

using quant::machina::GetDenseAttrFromTensorProtoAttr;
using quant::machina::GetIntTypeFromTFQint;
using quant::machina::IsTFQintType;
using quant::machina::IsTFUniformQuantizedOp;

#define GEN_PASS_DEF_CONVERTTFQUANTTYPES
#include "machina/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h.inc"

// TODO: b/290366702 - Temporarily added metrics for debugging.
auto *mlir_tf_quant_op_count = ::machina::monitoring::Counter<1>::New(
    "/machina/core/tf2xla/tf_quant_op_count" /*metric_name*/,
    "Counts the number of ops that has qint types" /*metric description*/,
    "op_name" /*metric label*/);

// Returns whether a type is illegal. Here we consider TF qint types illegal.
// See pass description in passes.td for more info about how illegal types are
// treated in this pass.
bool IsIllegalType(Type type) {
  return IsTFQintType(getElementTypeOrSelf(type));
}

// Get the corresponding int type from TF qint types.
// If input is not TF qint types, returns the original type.
Type ToLegalType(Type type) {
  if (IsTFQintType(type)) return GetIntTypeFromTFQint(type);
  if (auto shaped = mlir::dyn_cast<ShapedType>(type)) {
    Type elem = shaped.getElementType();
    if (IsTFQintType(elem)) return shaped.clone(ToLegalType(elem));
  }
  return type;
}

bool IsQintToIntCast(Operation *op) {
  auto cast_op = toolchain::dyn_cast<TF::CastOp>(op);
  return cast_op && IsIllegalType(cast_op.getX().getType()) &&
         ToLegalType(cast_op.getX().getType()) == cast_op.getY().getType();
}

bool IsIntToQintCast(Operation *op) {
  auto cast_op = toolchain::dyn_cast<TF::CastOp>(op);
  return cast_op && IsIllegalType(cast_op.getY().getType()) &&
         ToLegalType(cast_op.getY().getType()) == cast_op.getX().getType();
}

// Check if an op result value is consumed by qint -> int TF Cast OP.
bool IsQintValueQintToIntCast(Value v) {
  if (!IsIllegalType(v.getType())) {
    return true;
  }
  if (v.getUsers().empty()) {
    return false;
  }
  return toolchain::all_of(v.getUsers(), [&](OpOperand operand) {
    return IsQintToIntCast(operand.getOwner());
  });
}

// Check if an op operand value is defined by int -> qint TF Cast OP.
bool IsQintValueDefinedByIntToQintCast(Value v) {
  if (!IsIllegalType(v.getType())) {
    return true;
  }
  if (!v.getDefiningOp() || !toolchain::isa<TF::CastOp>(v.getDefiningOp())) {
    return false;
  }
  return IsIntToQintCast(v.getDefiningOp());
}

bool IsTFUniformQuantizedOpLegal(Operation *op) {
  // UniformQuantized Ops are considered legal if its qint operands and
  // results are connected to TF CastOp.
  return op && toolchain::all_of(op->getResults(), IsQintValueQintToIntCast) &&
         toolchain::all_of(op->getOperands(), IsQintValueDefinedByIntToQintCast);
}

bool IsCastOpLegal(TF::CastOp cast_op) {
  // Consider qint <-> qint casts illegal.
  if (IsIllegalType(cast_op.getSrcT()) && IsIllegalType(cast_op.getDstT())) {
    return false;
  }
  // Consider CastOp illegal if either of its Src/Dst type is qint and is
  // connected to a non-UQ op.
  if (IsIllegalType(cast_op.getSrcT()) &&
      !(cast_op.getX().getDefiningOp() &&
        IsTFUniformQuantizedOp(cast_op.getX().getDefiningOp()))) {
    return false;
  }
  if (IsIllegalType(cast_op.getDstT()) &&
      !IsTFUniformQuantizedOp(*cast_op.getY().getUsers().begin())) {
    return false;
  }
  return true;
}

class TFQuantTypeConverter : public TypeConverter {
 public:
  TFQuantTypeConverter() {
    addConversion([](Type type) -> Type {
      return IsIllegalType(type) ? ToLegalType(type) : type;
    });
  }
};

// An Op is illegal iff it is non-UQ op and it contains qint types.
class TFQuantTypeConversionTarget : public ConversionTarget {
 public:
  explicit TFQuantTypeConversionTarget(MLIRContext &ctx,
                                       TFQuantTypeConverter &converter)
      : ConversionTarget(ctx), converter_(converter) {
    markUnknownOpDynamicallyLegal([this](Operation *op) {
      // Consider UQ op legal if it has a CastOp next to the qint input/output.
      if (IsTFUniformQuantizedOp(op)) {
        return IsTFUniformQuantizedOpLegal(op);
      } else if (auto cast_op = toolchain::dyn_cast<TF::CastOp>(op)) {
        return IsCastOpLegal(cast_op);
      } else if (auto const_op = toolchain::dyn_cast<TF::ConstOp>(op)) {
        return !IsIllegalType(const_op.getOutput().getType());
      }
      // The FuncOp type can contain types that the op's operand and result
      // types do not contain.
      if (auto func = dyn_cast<func::FuncOp>(op)) {
        if (!converter_.isSignatureLegal(func.getFunctionType())) return false;
      }
      return converter_.isLegal(op);
    });
  }

 private:
  TFQuantTypeConverter &converter_;
};

class TFQuantTypePattern : public ConversionPattern {
 public:
  TFQuantTypePattern(MLIRContext *ctx, TypeConverter &converter)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // This pattern only handle non-UQ, non-const ops.
    if (IsTFUniformQuantizedOp(op) || toolchain::isa<TF::ConstOp>(op)) {
      return failure();
    }

    // Update the results.
    toolchain::SmallVector<Type, 4> new_results;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                new_results)))
      return failure();

    // Update the regions. The dialect conversion framework wants new regions to
    // be created and updated, rather than updating the old op. Thus we use an
    // OperationState so we can add regions to the new up.
    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         new_results, op->getAttrs(), op->getSuccessors());
    for (Region &region : op->getRegions()) {
      auto new_region = std::make_unique<Region>(op);
      rewriter.inlineRegionBefore(region, *new_region, new_region->begin());
      if (failed(rewriter.convertRegionTypes(new_region.get(),
                                             *getTypeConverter()))) {
        return failure();
      }
      state.addRegion(std::move(new_region));
    }
    rewriter.replaceOp(op, rewriter.create(state)->getResults());

    // TODO: b/290366702 - Temporarily added metrics for debugging.
    mlir_tf_quant_op_count->GetCell(std::string(op->getName().getStringRef()))
        ->IncrementBy(1);
    return success();
  }
};

// This pattern adds qint <-> int Cast to all qint operands and results for UQ
// ops.
class TFUniformQuantizedOpsPattern : public ConversionPattern {
 public:
  explicit TFUniformQuantizedOpsPattern(MLIRContext *ctx)
      : ConversionPattern(MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // This pattern only handle UQ ops.
    if (!IsTFUniformQuantizedOp(op)) {
      return failure();
    }

    // Add CastOp int->qint before an input operand only when it original type
    // is qint and its defining op is not already an int->qint CastOp.
    toolchain::SmallVector<Value, 4> new_operands;
    for (int i = 0; i < operands.size(); ++i) {
      Type orig_op_type = op->getOperandTypes()[i];
      if (IsIllegalType(orig_op_type) &&
          !IsQintValueDefinedByIntToQintCast(op->getOperand(i))) {
        new_operands.push_back(rewriter.create<TF::CastOp>(
            op->getLoc(), orig_op_type, operands[i]));
      } else {
        new_operands.push_back(operands[i]);
      }
    }

    // Create a new UQ op.
    OperationState state(op->getLoc(), op->getName().getStringRef(),
                         new_operands, op->getResultTypes(), op->getAttrs(),
                         op->getSuccessors());
    Operation *new_op = rewriter.create(state);
    toolchain::SmallVector<Value, 4> new_results = new_op->getResults();

    // Add qint->int CastOp after output result if its original type is qint and
    // its users are not all qint->int CastOps.
    for (int i = 0; i < new_results.size(); ++i) {
      Value &result = new_results[i];
      if (IsIllegalType(result.getType()) &&
          !IsQintValueQintToIntCast(op->getResult(i))) {
        result = rewriter.create<TF::CastOp>(
            op->getLoc(), ToLegalType(result.getType()), result);
      }
      // If the result is already consumed by qint->int CastOp, manually replace
      // its use by the new UQ op. This is because such CastOp is already legal,
      // it will not go through any conversion pattern later. Without this, that
      // CastOp will still be consuming the original UQ op and cause errors.
      op->getResult(i).replaceUsesWithIf(
          new_op->getResult(i), [](OpOperand &operand) {
            return IsQintToIntCast(operand.getOwner());
          });
    }
    rewriter.replaceOp(op, new_results);
    return success();
  }
};

class TFConstOpQuantToIntPattern : public OpConversionPattern<TF::ConstOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::ConstOp op, TF::ConstOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!IsIllegalType(op.getOutput().getType())) return failure();
    TF::TensorProtoAttr tensor_proto_attr;
    if (!matchPattern(op.getOperation(), m_Constant(&tensor_proto_attr))) {
      return rewriter.notifyMatchFailure(op, "operand must be constant.");
    }
    auto dense_attr_or = GetDenseAttrFromTensorProtoAttr(
        tensor_proto_attr.getValue(),
        mlir::dyn_cast<TensorType>(ToLegalType(op.getOutput().getType())));
    if (failed(dense_attr_or)) {
      op->emitError("failed to get DenseElementAttr.");
      return failure();
    }

    rewriter.replaceOpWithNewOp<TF::ConstOp>(
        op, ToLegalType(op.getOutput().getType()), *dense_attr_or);
    return success();
  }
};

struct ConvertTFQuantTypes
    : public impl::ConvertTFQuantTypesBase<ConvertTFQuantTypes> {
  void runOnOperation() override;
};

void ConvertTFQuantTypes::runOnOperation() {
  TFQuantTypeConverter converter;
  RewritePatternSet patterns(&getContext());
  patterns.add<TFQuantTypePattern>(&getContext(), converter);
  patterns.add<TFConstOpQuantToIntPattern, TFUniformQuantizedOpsPattern>(
      &getContext());
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 converter);
  TFQuantTypeConversionTarget target(getContext(), converter);
  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateConvertTFQuantTypesPass() {
  return std::make_unique<ConvertTFQuantTypes>();
}

}  // namespace mlir::quant::stablehlo
