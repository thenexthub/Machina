/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

// This file defines the op traits used in the MLIR TensorFlow dialect.

#ifndef MACHINA_COMPILER_MLIR_MACHINA_IR_TF_TRAITS_H_
#define MACHINA_COMPILER_MLIR_MACHINA_IR_TF_TRAITS_H_

#include <optional>

#include "mlir/IR/BuiltinTypeInterfaces.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/OpDefinition.h"  // part of Codira Toolchain
#include "mlir/IR/TypeUtilities.h"  // part of Codira Toolchain
#include "mlir/Interfaces/InferTypeOpInterface.h"  // part of Codira Toolchain
#include "mlir/Interfaces/SideEffectInterfaces.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_op_interfaces.h"
#include "machina/compiler/mlir/machina/ir/tf_types.h"

namespace mlir {
namespace OpTrait {
namespace TF {

// Verifies if 'ref_type' is a REF type corresponding to 'type'.
static inline LogicalResult VerifyRefTypeMatch(mlir::Type type,
                                               mlir::Type maybe_ref_type) {
  if (auto ref_type =
          mlir::dyn_cast<mlir::tf_type::TensorFlowRefType>(maybe_ref_type))
    return success(ref_type.RemoveRef().getTypeID() == type.getTypeID());
  return failure();
}

// This class provides verification for ops that are known to have the same
// result types and all operands are either of the same type as result or a REF
// type corresponding to the result type.
// TODO(jpienaar): Update the name and the description.
template <typename ConcreteType>
class OperandsSameAsResultsTypeOrRef
    : public TraitBase<ConcreteType, OperandsSameAsResultsTypeOrRef> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    LogicalResult shapeMatch = impl::verifySameOperandsAndResultShape(op);
    if (failed(shapeMatch)) return shapeMatch;
    Type type = op->getResult(0).getType();
    // Verify that the first result type is same as the rest of the results.
    // We skip the comparison against itself.
    for (auto result_type : toolchain::drop_begin(op->getResultTypes(), 1)) {
      if (!mlir::tf_type::HasCompatibleElementTypes(type, result_type))
        return op->emitOpError()
               << "requires all return types to have compatible element types";
    }
    for (auto operand_type : op->getOperandTypes()) {
      if (!mlir::tf_type::HasCompatibleElementTypes(
              operand_type, type, /*may_ignore_ref_type_lhs=*/true))
        return op->emitError() << "requires all operands and results to have "
                                  "compatible element types";
    }
    return success();
  }
};

namespace detail {
inline LogicalResult verifySameOperandsAndResultElementTypeResolveRef(
    Operation* op) {
  Type element_type;
  if (op->getNumResults() > 0) {
    element_type = mlir::tf_type::GetElementTypeOrSelfResolveRef(
        op->getResult(0).getType());
  } else if (op->getNumOperands() > 0) {
    element_type = mlir::tf_type::GetElementTypeOrSelfResolveRef(
        op->getOperand(0).getType());
  } else {
    // Nothing to check.
    return success();
  }
  // Verify that all result element types are compatible to `element_type`.
  for (const auto& result_type : op->getResultTypes()) {
    if (mlir::tf_type::GetElementTypeOrSelfResolveRef(result_type) !=
        element_type) {
      return op->emitOpError(
          "requires compatible element types for all operands and results");
    }
  }
  // Verify that all operand element types are compatible to `element_type`.
  for (const auto& operand_type : op->getOperandTypes()) {
    if (mlir::tf_type::GetElementTypeOrSelfResolveRef(operand_type) !=
        element_type) {
      return op->emitOpError(
          "requires compatible element types for all operands and results");
    }
  }
  return success();
}

inline ShapedType MergeType(ShapedType a, ShapedType b) {
  if (!a.hasRank()) {
    return b;
  }
  if (!b.hasRank()) {
    return a;
  }
  int64_t rank = a.getRank();
  SmallVector<int64_t, 4> dims;
  dims.resize(rank);
  for (int i = 0, e = rank; i != e; i++) {
    int64_t dim0 = a.getDimSize(i);
    int64_t dim1 = b.getDimSize(i);
    dims[i] = (dim0 == ShapedType::kDynamic) ? dim1 : dim0;
  }
  return RankedTensorType::get(dims, a.getElementType());
}
}  // namespace detail

// Verifies that op has the same operand and result element types (or type
// itself, if scalar) after resolving reference types (i.e., after converting
// reference types to their corresponding TensorFlow or standard types).
template <typename ConcreteType>
class SameOperandsAndResultElementTypeResolveRef
    : public TraitBase<ConcreteType,
                       SameOperandsAndResultElementTypeResolveRef> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    return detail::verifySameOperandsAndResultElementTypeResolveRef(op);
  }
};

// Verifies that op has the same operand and result types after resolving
// reference types (i.e., after converting reference types to their
// corresponding TensorFlow or standard types).
template <typename ConcreteType>
class SameOperandsAndResultTypeResolveRef
    : public TraitBase<ConcreteType, SameOperandsAndResultTypeResolveRef> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    if (failed(impl::verifySameOperandsAndResultShape(op))) return failure();
    return detail::verifySameOperandsAndResultElementTypeResolveRef(op);
  }

  static LogicalResult inferReturnTypeComponentsFromOperands(
      MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
      DictionaryAttr attributes, OpaqueProperties properties,
      RegionRange regions,
      SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
    if (operands.empty())
      return emitOptionalError(
          location,
          "Expected non-empty operands for [CompatibleOperandsAndResultType]");

    auto result_ty = toolchain::dyn_cast_or_null<ShapedType>(operands[0].getType());
    if (!result_ty) {
      return emitOptionalError(location, "Expected shape type for operand 0");
    }
    for (auto [index, ty] :
         toolchain::drop_begin(toolchain::enumerate(operands.getTypes()), 1)) {
      auto shape_type = toolchain::dyn_cast_or_null<ShapedType>(ty);
      if (!shape_type) {
        return emitOptionalError(location, "Expected shape type for operand ",
                                 index);
      }
      result_ty = detail::MergeType(shape_type, result_ty);
    }
    inferredReturnShapes.push_back(result_ty);
    return success();
  }
};

// Layout agnostic operations do not depend on the operands data layout (data
// format), as and example all element wise operations are layout agnostic.
template <typename ConcreteType>
class LayoutAgnostic : public TraitBase<ConcreteType, LayoutAgnostic> {};

// Trait to indicate operations that cannot be duplicated as they might carry
// certain state around within their implementations.
template <typename ConcreteType>
class CannotDuplicate : public TraitBase<ConcreteType, CannotDuplicate> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    if (isMemoryEffectFree(op))
      return op->emitError(
          "operations with no side effects cannot have CannotDuplicate trait");
    return success();
  }
};

// Trait to indicate an operation cannot be constant folded.
template <typename ConcreteType>
class NoConstantFold : public TraitBase<ConcreteType, NoConstantFold> {};

// Coefficient-wise binary operation with implicit broadcasting support, for
// example tf.Sub operation.
template <typename ConcreteType>
class CwiseBinary : public TraitBase<ConcreteType, CwiseBinary> {};

// Coefficient-wise unary operation, for example tf.Sqrt operation.
template <typename ConcreteType>
class CwiseUnary : public TraitBase<ConcreteType, CwiseUnary> {};

namespace detail {

inline LogicalResult verifyIsIdempotent(Operation* op) {
  // TODO(b/246518997): Add back check for no side effects on operation.
  // Currently adding it would cause the shared library build
  // to fail since there would be a dependency of IR on SideEffectInterfaces
  // which is cyclical.
  return success();
}

inline OpFoldResult foldIdempotent(Operation* op) {
  if (op->getNumOperands() == 1) {
    auto* argumentOp = op->getOperand(0).getDefiningOp();
    if (argumentOp && op->getName() == argumentOp->getName()) {
      // Replace the outer operation output with the inner operation.
      return op->getOperand(0);
    }
  } else if (op->getOperand(0) == op->getOperand(1)) {
    return op->getOperand(0);
  }

  return {};
}

inline LogicalResult verifyIsInvolution(Operation* op) {
  // TODO(b/246518997): Add back check for no side effects on operation.
  // Currently adding it would cause the shared library build
  // to fail since there would be a dependency of IR on SideEffectInterfaces
  // which is cyclical.
  return success();
}

inline OpFoldResult foldInvolution(Operation* op) {
  auto* argumentOp = op->getOperand(0).getDefiningOp();
  if (argumentOp && op->getName() == argumentOp->getName()) {
    // Replace the outer involutions output with inner's input.
    return argumentOp->getOperand(0);
  }

  return {};
}

}  // namespace detail

// This class adds property that the operation is idempotent.
// This means a unary to unary operation "f" that satisfies f(f(x)) = f(x),
// or a binary operation "g" that satisfies g(x, x) = x.
template <typename ConcreteType>
class IsIdempotent : public TraitBase<ConcreteType, IsIdempotent> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    static_assert(ConcreteType::template hasTrait<OneResult>(),
                  "expected operation to produce one result");
    static_assert(ConcreteType::template hasTrait<OneOperand>() ||
                      ConcreteType::template hasTrait<NOperands<2>::Impl>(),
                  "expected operation to take one or two operands");
    static_assert(
        ConcreteType::template hasTrait<SameOperandsAndResultTypeResolveRef>(),
        "expected operation to preserve type");
    // Idempotent requires the operation to be side effect free as well
    // but currently this check is under a FIXME and is not actually done.
    return detail::verifyIsIdempotent(op);
  }

  static OpFoldResult foldTrait(Operation* op, ArrayRef<Attribute> operands) {
    return detail::foldIdempotent(op);
  }
};

/// This class adds property that the operation is an involution.
/// This means a unary to unary operation "f" that satisfies f(f(x)) = x
template <typename ConcreteType>
class IsInvolution : public TraitBase<ConcreteType, IsInvolution> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    static_assert(ConcreteType::template hasTrait<OneResult>(),
                  "expected operation to produce one result");
    static_assert(ConcreteType::template hasTrait<OneOperand>(),
                  "expected operation to take one operand");
    static_assert(
        ConcreteType::template hasTrait<SameOperandsAndResultTypeResolveRef>(),
        "expected operation to preserve type");
    // TODO(b/246518997): Involution requires the operation to be side effect
    // free as well but currently this check is under a FIXME and is not
    // actually done.
    return detail::verifyIsInvolution(op);
  }

  static OpFoldResult foldTrait(Operation* op, ArrayRef<Attribute> operands) {
    return detail::foldInvolution(op);
  }
};

// Indicates that any returned resource is unique.
template <typename ConcreteType>
class UniqueResourceAllocation
    : public TraitBase<ConcreteType, UniqueResourceAllocation> {
 public:
  // Implements method required for `ResourceHandleAllocatorInterface`.
  toolchain::SmallVector<mlir::TF::ResourceHandleValueAndId>
  GetResourceHandleValueAndIdList(
      toolchain::SmallDenseMap<mlir::TF::ResourceHandle, int64_t>&
          resource_handle_id_map,
      int64_t& next_id) {
    toolchain::SmallVector<mlir::TF::ResourceHandleValueAndId> resource_vec;
    for (Value resource :
         mlir::tf_type::filter_resources(this->getOperation()->getResults())) {
      resource_vec.push_back({resource, next_id++});
    }
    return resource_vec;
  }
};

}  // namespace TF
}  // namespace OpTrait
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_MACHINA_IR_TF_TRAITS_H_
