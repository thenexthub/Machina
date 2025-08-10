/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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
#include <string>
#include <utility>

#include "toolchain/ADT/DenseSet.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypeInterfaces.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Diagnostics.h"  // part of Codira Toolchain
#include "mlir/IR/IRMapping.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "stablehlo/dialect/Base.h"  // from @stablehlo
#include "machina/compiler/mlir/op_or_arg_name_mapper.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tpu_embedding_ops_registry.h"
#include "machina/compiler/mlir/tf2xla/transforms/legalization_op_config.h"
#include "machina/compiler/mlir/tf2xla/transforms/legalize_tf_with_tf2xla_passes.h"
#include "machina/compiler/mlir/tf2xla/transforms/passes.h"
#include "machina/compiler/mlir/tf2xla/transforms/tf2xla_rewriter.h"
#include "machina/compiler/tf2xla/xla_compilation_device.h"
#include "machina/compiler/tf2xla/xla_context.h"
#include "machina/compiler/tf2xla/xla_expression.h"
#include "machina/compiler/tf2xla/xla_helpers.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/tsl/platform/env.h"
#include "machina/xla/tsl/platform/status.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/core/common_runtime/device.h"
#include "machina/core/common_runtime/device_factory.h"
#include "machina/core/common_runtime/device_mgr.h"
#include "machina/core/common_runtime/process_function_library_runtime.h"
#include "machina/core/framework/allocator.h"
#include "machina/core/framework/function.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/framework/node_properties.h"
#include "machina/core/framework/op.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/resource_mgr.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/types.h"
#include "machina/core/framework/types.pb.h"

namespace mlir {
namespace mhlo {

namespace {

// Returns true if the given type is a ranked tensor type with static or bounded
// dimensions.
bool IsBounded(Type ty) {
  auto ranked_ty = mlir::dyn_cast<RankedTensorType>(ty);
  if (!ranked_ty) return false;

  if (ranked_ty.hasStaticShape()) return true;

  auto bounds = hlo::encodingToBounds(ranked_ty.getEncoding());
  if (bounds.empty()) return false;

  for (int i = 0; i < ranked_ty.getRank(); ++i) {
    if (ranked_ty.isDynamicDim(i) && bounds[i] == ShapedType::kDynamic) {
      return false;
    }
  }
  return true;
}

bool HasSymbolRefAttr(Operation* op) {
  for (const auto& attr : op->getAttrs()) {
    Attribute attr_value = attr.getValue();
    if (mlir::isa<SymbolRefAttr>(attr_value)) {
      return true;
    } else if (auto array_attr = mlir::dyn_cast<ArrayAttr>(attr_value)) {
      if (!array_attr.empty() &&
          mlir::isa<SymbolRefAttr>(*array_attr.begin())) {
        return true;
      }
    }
  }
  return false;
}

class Tf2XlaRewritePattern : public ConversionPattern {
 public:
  explicit Tf2XlaRewritePattern(MLIRContext* ctx, TypeConverter& converter,
                                const std::string& device_type,
                                bool prefer_tf2xla)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
        device_type_(device_type),
        prefer_tf2xla_(prefer_tf2xla) {}

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    // This pattern is a conversion pattern because we want to specify a type
    // converter. However, this pattern still uses the original op's operands
    // while creating the ops so make sure there aren't any type changes between
    // the original op operands and the operands during the conversion.
    for (auto&& [old_val, new_val] : toolchain::zip(op->getOperands(), operands)) {
      if (old_val.getType() != new_val.getType()) return failure();
    }

    auto abstractOp = op->getRegisteredInfo();
    if (!abstractOp) return failure();

    if (!(hlo::IsOpAllowedTf2xlaFallback(abstractOp->getTypeID()) ||
          (prefer_tf2xla_ &&
           hlo::IsOpAllowedTf2xlaPreferred(abstractOp->getTypeID())))) {
      return failure();
    }

    return hlo::Tf2XlaRewriter::RewriteOp(op, rewriter, device_type_);
  }

 private:
  std::string device_type_;
  bool prefer_tf2xla_;
  bool use_tf2xla_hlo_importer_;
};

bool ShouldRefineTypeTo(Type original_ty, Type updated_ty) {
  auto updated = mlir::dyn_cast<ShapedType>(updated_ty);
  auto original = mlir::dyn_cast<ShapedType>(original_ty);

  // Both types must be shaped types.
  if (!original || !updated) return false;

  // Element types must match.
  if (original.getElementType() != updated.getElementType()) return false;

  // If the updated type doesn't have a rank, then it can't be a more refined
  // type.
  if (!updated.hasRank()) return false;

  // If the original type doesn't have a rank, then refine as the updated type
  // has a rank.
  if (!original.hasRank()) return true;

  // Both types must have the same rank.
  if (original.getRank() != updated.getRank()) return false;

  // Refine if the updated type is bounded.
  return IsBounded(updated);
}

// Propagates more refined type by cloning op using the new operands. This
// allows all rewrite patterns that requires refined types to work without
// requiring a rewrite to the conversion pattern. Declarative rewrite pattern
// (DRR) doesn't even support conversion patterns with TableGen.
class TypePropagator : public ConversionPattern {
 public:
  explicit TypePropagator(MLIRContext* ctx)
      : ConversionPattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    // This could be generalized to other ops as needs arise. We could even
    // remove this restriction altogether except for the terminators that
    // require function signature change and shouldn't be
    if (op->getName().getDialectNamespace() !=
        TF::TensorFlowDialect::getDialectNamespace())
      return failure();

    // Refining types may have implications to the attached regions or symbol
    // references so do not update such ops.
    if (!op->getRegions().empty() || HasSymbolRefAttr(op)) return failure();

    IRMapping mapper;
    bool has_type_change = false;
    for (auto [original, updated] : toolchain::zip(op->getOperands(), operands)) {
      Type original_ty = original.getType();
      Type updated_ty = updated.getType();
      if (original_ty != updated_ty) has_type_change = true;

      if (!ShouldRefineTypeTo(original_ty, updated_ty)) return failure();
      mapper.map(original, updated);
    }
    if (!has_type_change) return failure();

    Operation* cloned_op = rewriter.clone(*op, mapper);
    rewriter.replaceOp(op, cloned_op->getResults());
    return success();
  }
};

}  // end namespace

Tf2XlaTypeConverter::Tf2XlaTypeConverter() {
  // Currently, we don't do any type conversions. Any TensorFlow op with a type
  // that is not supported in MHLO will fail conversion. Quantized types are
  // going to handled separately so we don't need to handle those.
  addConversion([](Type ty) { return ty; });

  // This materialization is helpful in cases where we have more refined types
  // after conversion to mhlo compared to the original type in TF. For example,
  // a TF op with result type tensor<*xf32> will have a bounded type after
  // fallback legalization.
  auto cast_value = [&](OpBuilder& builder, Type result_type, ValueRange inputs,
                        Location loc) -> Value {
    return builder.create<mlir::tensor::CastOp>(loc, result_type,
                                                inputs.front());
  };
  addSourceMaterialization(cast_value);
}

void PopulateLegalizeTfWithTf2XlaPatterns(toolchain::StringRef device_type,
                                          RewritePatternSet& patterns,
                                          MLIRContext* ctx,
                                          Tf2XlaTypeConverter& converter,
                                          bool prefer_tf2xla) {
  patterns.add<TypePropagator>(ctx);
  patterns.add<Tf2XlaRewritePattern>(ctx, converter, device_type.str(),
                                     prefer_tf2xla);
}

}  // end namespace mhlo
}  // end namespace mlir
