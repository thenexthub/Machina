/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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
#include <optional>
#include <vector>

#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/Support/Casting.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo

namespace mlir::odml {

namespace {

// Helper class for parsing operands to a foldable operation.
class FoldAdaptor {
 public:
  // Returns std::nullopt if the operation cannot be folded.
  static std::optional<FoldAdaptor> Create(Operation* operation) {
    auto foldable_opr = [](Value val) -> bool {
      return !toolchain::isa<BlockArgument>(val) &&
             toolchain::isa<stablehlo::ConstantOp>(val.getDefiningOp());
    };
    if (!toolchain::all_of(operation->getOperands(), foldable_opr)) {
      return std::nullopt;
    }
    return FoldAdaptor(operation);
  }

  // Gets a list of ElementsAttr behind each constant operand.
  toolchain::SmallVector<ElementsAttr> OperandData() {
    toolchain::SmallVector<ElementsAttr> res;
    res.reserve(operation_->getNumOperands());
    for (auto opr : operation_->getOperands()) {
      auto op = toolchain::dyn_cast<stablehlo::ConstantOp>(opr.getDefiningOp());
      res.push_back(op.getValue());
    }
    return res;
  }

  // Gets a pointer to the operation to be folded.
  Operation* Op() { return operation_; }

 private:
  explicit FoldAdaptor(Operation* operation) : operation_(operation) {}
  Operation* const operation_;
};

// APSInt provides operators which APInt does not, so allow for converting
// to APSInt for computation. Only APInts can be directly read from
// element attributes.
static const APFloat& AddSign(const APFloat& v) { return v; }
static APSInt AddSign(const APInt& v) { return APSInt(v); }

template <typename ResultType>
static LogicalResult FoldDivOpInternal(stablehlo::DivOp op,
                                       PatternRewriter& rewriter) {
  auto adaptor = FoldAdaptor::Create(op);
  if (!adaptor.has_value()) {
    return failure();
  }
  auto const_oprs = adaptor.value().OperandData();

  const bool lhs_splat = const_oprs[0].isSplat();
  const bool rhs_splat = const_oprs[1].isSplat();

  auto lhs_vals = const_oprs[0].getValues<ResultType>();
  auto rhs_vals = const_oprs[1].getValues<ResultType>();
  const auto num_results = std::max(lhs_vals.size(), rhs_vals.size());
  std::vector<ResultType> res;
  res.reserve(num_results);

  auto lhs_start = lhs_vals.begin();
  auto rhs_start = rhs_vals.begin();

  for (int i = 0; i < num_results; ++i) {
    auto lhs_val = lhs_splat ? *lhs_start : *(lhs_start++);
    auto rhs_val = rhs_splat ? *rhs_start : *(rhs_start++);
    auto signed_lhs_val = AddSign(lhs_val);
    auto signed_rhs_val = AddSign(rhs_val);
    if (signed_rhs_val.isZero()) {
      return failure();
    }
    res.push_back(signed_lhs_val / signed_rhs_val);
  }

  auto res_attr = DenseElementsAttr::get(
      mlir::cast<RankedTensorType>(const_oprs[0].getType()), res);
  rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(adaptor.value().Op(),
                                                     res_attr);
  return success();
}

static LogicalResult FoldDivOp(stablehlo::DivOp op, PatternRewriter& rewriter) {
  auto etype = op.getType().getElementType();
  if (mlir::isa<FloatType>(etype)) {
    return FoldDivOpInternal<APFloat>(op, rewriter);
  }
  if (mlir::isa<IntegerType>(etype)) {
    return FoldDivOpInternal<APInt>(op, rewriter);
  }
  return failure();
}
}  // namespace

void PopulateFolderPatterns(RewritePatternSet& patternSet) {
  patternSet.add(FoldDivOp);
}

}  // namespace mlir::odml
