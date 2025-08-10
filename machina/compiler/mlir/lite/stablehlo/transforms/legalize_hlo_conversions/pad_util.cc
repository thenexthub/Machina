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
#include "machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/pad_util.h"

#include <cstdint>

#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypeInterfaces.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/op_util_common.h"
#include "machina/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::odml {

ShapedType GetPaddingAttrType(mhlo::PadOp op) {
  return op.getEdgePaddingLow().getType();
}

DenseIntElementsAttr SliceStartFromNegPadLows(mhlo::PadOp op) {
  auto vals = UnrollI64Splat(op.getEdgePaddingLow());
  auto starts = toolchain::map_range(
      vals, [](auto v) -> int64_t { return (v >= 0) ? 0 : -1 * v; });
  return DenseIntElementsAttr::get(GetPaddingAttrType(op),
                                   toolchain::to_vector(starts));
}

DenseIntElementsAttr SliceEndFromNegPadHighs(mhlo::PadOp op) {
  auto vals = UnrollI64Splat(op.getEdgePaddingHigh());
  auto zip = toolchain::zip(vals, op.getOperand().getType().getShape());
  auto ends = toolchain::map_range(zip, [](auto it) -> int64_t {
    return (std::get<0>(it) >= 0) ? std::get<1>(it)
                                  : std::get<1>(it) + std::get<0>(it);
  });
  return DenseIntElementsAttr::get(GetPaddingAttrType(op),
                                   toolchain::to_vector(ends));
}

DenseIntElementsAttr ReplaceNegsWithZero(DenseElementsAttr data) {
  auto vals = UnrollI64Splat(data);
  auto res =
      toolchain::map_range(vals, [](auto v) -> int64_t { return (v < 0) ? 0 : v; });
  return DenseIntElementsAttr::get(data.getType(), toolchain::to_vector(res));
}

bool AnyNegativePads(mhlo::PadOp op) {
  auto is_neg = [](int64_t v) { return v < 0; };
  auto lows_data = UnrollI64Splat(op.getEdgePaddingLow());
  auto highs_data = UnrollI64Splat(op.getEdgePaddingHigh());
  return toolchain::any_of(lows_data, is_neg) || toolchain::any_of(highs_data, is_neg);
}

bool TrivialInterior(mhlo::PadOp op) {
  auto interior = op.getInteriorPadding();
  const bool trivial_splat =
      interior.isSplat() && interior.getSplatValue<int64_t>() == 0;
  const bool all_trivial = toolchain::all_of(interior.getValues<int64_t>(),
                                        [](auto v) { return v == 0; });
  return trivial_splat || all_trivial;
}

}  // namespace mlir::odml
