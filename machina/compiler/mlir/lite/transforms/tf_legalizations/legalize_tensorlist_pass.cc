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
#include "machina/compiler/mlir/lite/transforms/tf_legalizations/legalize_tensorlist_pass.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/Casting.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/IR/TypeUtilities.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"
#include "machina/compiler/mlir/lite/schema/schema_generated.h"
#include "machina/compiler/mlir/lite/utils/convert_type.h"
#include "machina/compiler/mlir/lite/utils/utils.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_ops_n_z.h"
#include "machina/compiler/mlir/machina/ir/tf_types.h"

namespace {
using ::mlir::MLIRContext;
using ::mlir::Operation;
using ::mlir::TensorType;
using ::mlir::Type;
using ::mlir::Value;
using ::mlir::TF::VariantType;
using ::mlir::TFL::ConstBytesAttr;

ConstBytesAttr SerializeOptionsToBytes(MLIRContext* context,
                                       const std::vector<uint8_t>& options) {
  std::string content;
  content.assign(reinterpret_cast<const char*>(options.data()), options.size());
  return ConstBytesAttr::get(context, content);
}

ConstBytesAttr CreateListReserveOptions(MLIRContext* context,
                                        tflite::TensorType element_type) {
  std::vector<uint8_t> options;
  options.push_back(element_type);
  return SerializeOptionsToBytes(context, options);
}

std::optional<Type> GetSingularVariantBaseType(Value val) {
  auto val_t = toolchain::dyn_cast_or_null<mlir::tf_type::VariantType>(
      mlir::getElementTypeOrSelf(val));
  if (!val_t) {
    return std::nullopt;
  }
  toolchain::ArrayRef<mlir::TensorType> subtypes = val_t.getSubtypes();
  if (subtypes.size() != 1) {
    return std::nullopt;
  }
  return subtypes[0].getElementType();
}

// Create an `ConstBytesAttr` which encodes the options
// for the `tf.custom` tensor list op to be created. If the given
// op is not a `tf.TensorList*` op, return empty, although this case
// should never be trigged in practice since patterns are only applied
// on `tf.TensorList*` ops.
std::optional<ConstBytesAttr> CustomOptions(MLIRContext* context,
                                            mlir::Operation* op) {
  if (auto reserve =
          toolchain::dyn_cast_or_null<mlir::TF::TensorListReserveOp>(op)) {
    tflite::TensorType tflite_type =
        tflite::ConvertTypeToTensorType(reserve.getElementDtype());

    return CreateListReserveOptions(context, tflite_type);
  }
  if (auto empty = toolchain::dyn_cast_or_null<mlir::TF::EmptyTensorListOp>(op)) {
    tflite::TensorType tflite_type =
        tflite::ConvertTypeToTensorType(empty.getElementDtype());

    return CreateListReserveOptions(context, tflite_type);
  }
  return {};
}

bool HasVariantInputOrOutput(Operation* op) {
  const bool has_variant_input = toolchain::any_of(op->getOperands(), [](Value val) {
    return toolchain::isa<VariantType>(
        toolchain::cast<mlir::TensorType>(val.getType()).getElementType());
  });
  const bool has_variant_output =
      toolchain::any_of(op->getResultTypes(), [](Type t) {
        return toolchain::isa<VariantType>(
            toolchain::cast<mlir::TensorType>(t).getElementType());
      });
  return has_variant_input || has_variant_output;
}

// There are 2 standard tf ops which are not TensorList ops that may take as
// input a tensorlist. These are tf.AddN and tf.ZeroesLike. Since the runtime
// implementation of a tensorlist are not compatible between tf and tflite
// we cannot use tflite tensorlist kernels until these cases are handled.
bool IsNonTensorListVariantOp(Operation* op) {
  return toolchain::isa<mlir::TF::ZerosLikeOp>(op) && HasVariantInputOrOutput(op);
}

}  // namespace

namespace mlir {
namespace TFL {
namespace {
#include "machina/compiler/mlir/lite/transforms/generated_legalize_tensorlist.inc"

struct ConvertTensorListPopBack
    : public OpRewritePattern<TF::TensorListPopBackOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::TensorListPopBackOp op,
                                PatternRewriter& rewriter) const override {
    // It is currently not possible to easily pack the output of a multi-result
    // op into an op with a single varidic output in `.td`.
    auto converted = rewriter.create<TFL::CustomOp>(
        op->getLoc(), op->getResultTypes(), op->getOperands(),
        "TensorListPopBack", TFL::ConstBytesAttr::get(getContext(), ""));
    rewriter.replaceOp(op, converted.getResults());
    return success();
  }
};

struct ConvertTensorListPushBack
    : public OpRewritePattern<TF::TensorListPushBackOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::TensorListPushBackOp op,
                                PatternRewriter& rewriter) const override {
    // It is currently not possible to easily pack the output of a multi-result
    // op into an op with a single varidic output in `.td`.
    auto converted = rewriter.create<TFL::CustomOp>(
        op->getLoc(), op->getResultTypes(), op->getOperands(),
        "TensorListPushBack", TFL::ConstBytesAttr::get(getContext(), ""));
    rewriter.replaceOp(op, converted.getResults());
    return success();
  }
};

struct ConvertVariantAddNOp : public OpRewritePattern<TF::AddNOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::AddNOp op,
                                PatternRewriter& rewriter) const override {
    if (!HasVariantInputOrOutput(op.getOperation())) {
      return failure();
    }
    auto converted = rewriter.create<TFL::CustomOp>(
        op->getLoc(), op->getResultTypes(), op->getOperands(), "VariantAddN",
        TFL::ConstBytesAttr::get(getContext(), ""));
    rewriter.replaceOp(op, converted.getResults());
    return success();
  }
};

bool IsOpSupported(mlir::Operation* op) {
  if (auto addn = toolchain::dyn_cast_or_null<TF::AddNOp>(op)) {
    if (HasVariantInputOrOutput(op)) {
      std::optional<mlir::Type> element_type =
          GetSingularVariantBaseType(op->getOperand(0));
      if (element_type.has_value()) {
        return element_type->isF32() || element_type->isInteger(32);
      }
    }
  }

  // Op is vacuously "supported" if it is not a tensorlist op.
  StringRef op_name = op->getName().getStringRef();
  if (!op_name.contains("TensorList")) return true;

  std::optional<mlir::Type> element_type = {};

  if (auto reserve = toolchain::dyn_cast_or_null<TF::TensorListReserveOp>(op)) {
    element_type = reserve.getElementDtype();
  }
  if (auto stack = toolchain::dyn_cast_or_null<TF::TensorListStackOp>(op)) {
    element_type = stack.getElementDtype();
  }
  if (auto set_item = toolchain::dyn_cast_or_null<TF::TensorListSetItemOp>(op)) {
    element_type = set_item.getElementDtype();
  }
  if (auto from_tensor =
          toolchain::dyn_cast_or_null<TF::TensorListFromTensorOp>(op)) {
    element_type = from_tensor.getElementDtype();
  }
  if (auto get_item = toolchain::dyn_cast_or_null<TF::TensorListGetItemOp>(op)) {
    element_type = get_item.getElementDtype();
  }
  if (auto empty = toolchain::dyn_cast_or_null<TF::EmptyTensorListOp>(op)) {
    element_type = empty.getElementDtype();
  }
  if (auto element_shape =
          toolchain::dyn_cast_or_null<TF::TensorListElementShapeOp>(op)) {
    element_type = GetSingularVariantBaseType(op->getOperand(0));
  }
  if (auto length = toolchain::dyn_cast_or_null<TF::TensorListLengthOp>(op)) {
    element_type = GetSingularVariantBaseType(op->getOperand(0));
  }
  if (auto pop_back = toolchain::dyn_cast_or_null<TF::TensorListPopBackOp>(op)) {
    element_type = pop_back.getElementDtype();
  }
  if (auto push_back = toolchain::dyn_cast_or_null<TF::TensorListPushBackOp>(op)) {
    element_type = push_back.getElementDtype();
  }

  if (!element_type.has_value()) return false;
  // TODO(b/288302706) add support for all types handled in the
  // `lower_static_tensor_list` pass.
  return element_type->isF32() || element_type->isInteger(64) ||
         element_type->isInteger(32) || element_type->isInteger(1);
}
}  // namespace

// Only legalize TensorFlow TensorList ops if all TensorList ops are supported
// natively.
void LegalizeTensorListPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  auto walk_res = module->walk([&](Operation* op) -> WalkResult {
    if (!IsOpSupported(op) || IsNonTensorListVariantOp(op)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walk_res.wasInterrupted()) {
    toolchain::errs() << "Tried legalizing to tfl custom tensorlist ops, but not "
                    "all can be supported."
                 << "\n";
    return;
  }
  RewritePatternSet patterns(&getContext());
  populateWithGenerated(patterns);
  patterns.add<ConvertTensorListPopBack>(&getContext());
  patterns.add<ConvertTensorListPushBack>(&getContext());
  patterns.add<ConvertVariantAddNOp>(&getContext());
  (void)applyPatternsGreedily(module, std::move(patterns));
}

}  // namespace TFL
}  // namespace mlir
