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

#include "machina/compiler/mlir/quantization/common/ir/QuantOps.h"

#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>

#include "toolchain/ADT/STLExtras.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OpDefinition.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/common/ir/QuantOpsDialect.cc.inc"

namespace mlir::quant::ir {

using mlir::quant::QuantizedType;

void TFQuantDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "machina/compiler/mlir/quantization/common/ir/QuantOps.cc.inc"
      >();
}

OpFoldResult StorageCastOp::fold(FoldAdaptor) {
  // Matches x -> [scast -> scast] -> y, replacing the second scast with the
  // value of x if the casts invert each other.
  auto srcScastOp = getArg().getDefiningOp<StorageCastOp>();
  if (!srcScastOp || srcScastOp.getArg().getType() != getType())
    return OpFoldResult();
  return srcScastOp.getArg();
}

/// The quantization specification should match the expressed type.
static bool isValidQuantizationSpec(Attribute quantSpec, Type expressed) {
  if (auto typeAttr = mlir::dyn_cast<TypeAttr>(quantSpec)) {
    Type spec = typeAttr.getValue();
    if (mlir::isa<TensorType, VectorType>(spec)) return false;

    // The spec should be either a quantized type which is compatible to the
    // expressed type, or a primitive type which is as same as the
    // (element type of) the expressed type.
    if (auto quantizedType = mlir::dyn_cast<QuantizedType>(spec))
      return quantizedType.isCompatibleExpressedType(expressed);

    if (auto tensorType = mlir::dyn_cast<TensorType>(expressed))
      return spec == tensorType.getElementType();

    if (auto vectorType = mlir::dyn_cast<VectorType>(expressed))
      return spec == vectorType.getElementType();
  }
  return false;
}

LogicalResult QuantizeRegionOp::verify() {
  // There are specifications for both inputs and outputs.
  if (getNumOperands() != getInputSpecs().size() ||
      getNumResults() != getOutputSpecs().size())
    return emitOpError(
        "has unmatched operands/results number and spec attributes number");

  // Verify that quantization specifications are valid.
  for (auto input : toolchain::zip(getOperandTypes(), getInputSpecs())) {
    Type inputType = std::get<0>(input);
    Attribute inputSpec = std::get<1>(input);
    if (!isValidQuantizationSpec(inputSpec, inputType)) {
      return emitOpError() << "has incompatible specification " << inputSpec
                           << " and input type " << inputType;
    }
  }

  for (auto result : toolchain::zip(getResultTypes(), getOutputSpecs())) {
    Type outputType = std::get<0>(result);
    Attribute outputSpec = std::get<1>(result);
    if (!isValidQuantizationSpec(outputSpec, outputType)) {
      return emitOpError() << "has incompatible specification " << outputSpec
                           << " and output type " << outputType;
    }
  }
  return success();
}

LogicalResult StatisticsOp::verify() {
  auto tensorArg = mlir::dyn_cast<TensorType>(getArg().getType());
  if (!tensorArg) return emitOpError("arg needs to be tensor type.");

  // Verify layerStats attribute.
  {
    auto layerStatsType = getLayerStats().getShapedType();
    if (!mlir::isa<FloatType>(layerStatsType.getElementType())) {
      return emitOpError("layerStats must have a floating point element type");
    }
    if (layerStatsType.getRank() != 1 || layerStatsType.getDimSize(0) != 2) {
      return emitOpError("layerStats must have shape [2]");
    }
  }
  // Verify axisStats (optional) attribute.
  if (getAxisStats()) {
    if (!getAxis()) return emitOpError("axis must be specified for axisStats");

    auto shape = tensorArg.getShape();
    auto argSliceSize =
        std::accumulate(std::next(shape.begin(), *getAxis()), shape.end(), 1,
                        std::multiplies<int64_t>());

    auto axisStatsType = getAxisStats()->getShapedType();
    if (!mlir::isa<FloatType>(axisStatsType.getElementType())) {
      return emitOpError("axisStats must have a floating point element type");
    }
    if (axisStatsType.getRank() != 2 || axisStatsType.getDimSize(1) != 2 ||
        axisStatsType.getDimSize(0) != argSliceSize) {
      return emitOpError(
          "axisStats must have shape [N,2] "
          "where N = the slice size defined by the axis dim");
    }
  }
  return success();
}

}  // namespace mlir::quant::ir

#define GET_OP_CLASSES
#include "machina/compiler/mlir/quantization/common/ir/QuantOps.cc.inc"
