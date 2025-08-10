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

#include "machina/compiler/mlir/machina/ir/tf_ops.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/strings/str_cat.h"
#include "toolchain/ADT/APFloat.h"
#include "toolchain/ADT/APInt.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/Sequence.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringExtras.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/ADT/StringSwitch.h"
#include "toolchain/ADT/iterator_range.h"
#include "toolchain/Support/Casting.h"
#include "toolchain/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/Traits.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Diagnostics.h"  // part of Codira Toolchain
#include "mlir/IR/DialectImplementation.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Matchers.h"  // part of Codira Toolchain
#include "mlir/IR/OpDefinition.h"  // part of Codira Toolchain
#include "mlir/IR/OpImplementation.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/IR/TypeUtilities.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Interfaces/FoldInterfaces.h"  // part of Codira Toolchain
#include "mlir/Interfaces/SideEffectInterfaces.h"  // part of Codira Toolchain
#include "mlir/Parser/Parser.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Transforms/InliningUtils.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_attributes.h"
#include "machina/compiler/mlir/machina/ir/tf_side_effects.h"
#include "machina/compiler/mlir/machina/ir/tf_structs.h"
#include "machina/compiler/mlir/machina/ir/tf_types.h"
#include "machina/core/common_runtime/inline_function_utils.h"
#include "machina/core/common_runtime/lower_function_call_inline_policy.h"
#include "machina/core/framework/op.h"
#include "machina/core/framework/op_def_builder.h"
#include "machina/core/platform/logging.h"
#include "machina/core/util/device_name_utils.h"
#include "machina/core/util/tensor_format.h"

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// TF Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {

struct TFConstantFoldInterface : public DialectFoldInterface {
  TFConstantFoldInterface(Dialect *dialect) : DialectFoldInterface(dialect) {}
  LogicalResult fold(Operation *op, ArrayRef<Attribute> operands,
                     SmallVectorImpl<OpFoldResult> &results) const final {
    return TensorFlowDialect::constantFold(op, operands, results);
  }
};

// Helper function that implements the multi-device inlining policy behavior
// for the inliner hook. In particular, for all function body nodes set unset
// placement attributes to match the function call node.
void MultiDeviceProcessInlinedCallBlocks(
    Operation *call, iterator_range<Region::iterator> inlinedBlocks) {
  using DeviceNameUtils = machina::DeviceNameUtils;

  // Duplicate of the logic in MultiDeviceFunctionBodyPlacer::BodyNodeDevice
  // LINT.IfChange
  auto device_id = StringAttr::get(call->getContext(), "device");
  auto caller_device = call->getAttrOfType<StringAttr>(device_id);
  if (!caller_device) return;

  DeviceNameUtils::ParsedName caller_parsed_device;
  if (!DeviceNameUtils::ParseFullName(caller_device.getValue().str(),
                                      &caller_parsed_device))
    return;

  MLIRContext *context = call->getContext();
  auto node_device = [&](Operation *n) -> StringAttr {
    auto device = n->getAttrOfType<StringAttr>(device_id);
    if (!device || device.getValue().empty()) return caller_device;

    DeviceNameUtils::ParsedName ndef_parsed_device;
    if (!DeviceNameUtils::ParseFullName(device.getValue().str(),
                                        &ndef_parsed_device))
      return device;
    DeviceNameUtils::MergeUnsetDevNames(&ndef_parsed_device,
                                        caller_parsed_device);
    return StringAttr::get(
        context, DeviceNameUtils::ParsedNameToString(ndef_parsed_device));
  };
  // LINT.ThenChange(../../../../core/common_runtime/inline_function_utils.cc)

  for (Block &block : inlinedBlocks) {
    block.walk([&](Operation *op) {
      if (op->getDialect() == call->getDialect())
        op->setAttr(device_id, node_device(op));
    });
  }
}

struct TFInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  // Returns if it's legal to inline 'callable' into the 'call', where 'call' is
  // a TF operation.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    // Skip inlining for TPUPartitionedCalls and RemoteCalls.
    if (isa<TPUPartitionedCallOp>(call)) return false;
    if (isa<RemoteCallOp>(call)) return false;
    // Maintain inlining for  `tf.function`s with jit_compile option.
    if (callable->hasAttr("tf._XlaMustCompile")) return true;
    auto noinline_attr_name = absl::StrCat("tf.", machina::kNoInlineAttr);
    if (auto noinline_attr =
            callable->getAttrOfType<BoolAttr>(noinline_attr_name))
      return !noinline_attr.getValue();
    return true;
  }

  // Returns if its legal to inline 'src' region into the 'dest' region
  // attached to a TF operation.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    // Allow inlining in regions attached to region based control flow
    // operations only if the src region is a single block region
    return isa<IfRegionOp, CaseRegionOp, WhileRegionOp>(dest->getParentOp()) &&
           toolchain::hasSingleElement(*src);
  }

  // Returns true if its legal to inline a TF operation `op` into the `dest`
  // region.
  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &) const final {
    // An op is legal to inline if either of the following conditions is true:
    // (a) Its legal to duplicate the Op.
    // (b) The Op is inside a single use function. If that function is inlined,
    //     post inlining, the function will be dead and eliminated from the IR.
    //     So there won't be any code duplication.
    // plus the function caller op can be replaced by inlined ops.
    return !wouldBeCloned || TensorFlowDialect::CanDuplicate(op);
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  // Attempts to materialize a conversion for a type mismatch between a call
  // from this dialect, and a callable region. This method should generate an
  // operation that takes 'input' as the only operand, and produces a single
  // result of 'resultType'. If a conversion can not be generated, nullptr
  // should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type result_type,
                                       Location conversion_loc) const final {
    if (!mlir::isa<TensorType>(result_type) ||
        !mlir::isa<TensorType>(input.getType()))
      return nullptr;
    return builder.create<TF::CastOp>(conversion_loc, result_type, input,
                                      /*truncate=*/builder.getBoolAttr(false));
  }

  void processInlinedCallBlocks(
      Operation *call,
      iterator_range<Region::iterator> inlinedBlocks) const final {
    bool has_lower_as_multi_device_function_attr = false;
    if (auto lower = call->getAttrOfType<BoolAttr>(
            machina::LowerFunctionalOpsConstants::
                kLowerAsMultiDeviceFunctionAttr))
      has_lower_as_multi_device_function_attr = lower.getValue();
    machina::FunctionCallInlinePolicy policy =
        machina::GetFunctionCallInlinePolicy(
            isa<PartitionedCallOp, StatefulPartitionedCallOp>(call),
            has_lower_as_multi_device_function_attr);

    if (policy == machina::FunctionCallInlinePolicy::kMultiDevicePlacer)
      return MultiDeviceProcessInlinedCallBlocks(call, inlinedBlocks);
  }
};
}  // end anonymous namespace

//===----------------------------------------------------------------------===//
// TF Dialect
//===----------------------------------------------------------------------===//

// Returns true if the op can be duplicated.
bool TensorFlowDialect::CanDuplicate(Operation *op) {
  // If the op is marked with the cannot duplicate trait, it cannot be
  // duplicated.
  if (op->hasTrait<OpTrait::TF::CannotDuplicate>()) return false;

  // If the op has no memory side effects, it can be duplicated.
  if (isMemoryEffectFree(op)) return true;

  // If the op is marked stateless using the `is_stateless` attribute, that
  // attribute determines if the op can be duplicated.
  if (auto is_stateless = op->getAttrOfType<BoolAttr>("is_stateless"))
    return is_stateless.getValue();

  // Assume ops can be duplicated if modelled.
  return op->isRegistered();
}

// TF dialect fallback for MemoryEffectOpInterface. The filtering for returning
// the interface is done in the return below and here it is empty as it is only
// returned for known not-stateful and unmodelled ops.
struct TensorFlowRegistryEffectInterfaceFallback
    : public MemoryEffectOpInterface::FallbackModel<
          TensorFlowRegistryEffectInterfaceFallback> {
  static bool classof(Operation *op) { return true; }
  void getEffects(
      Operation *op,
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
          &effects) const {}
};

void *TensorFlowDialect::getRegisteredInterfaceForOp(
    mlir::TypeID interface, mlir::OperationName opName) {
  if (interface == TypeID::get<mlir::MemoryEffectOpInterface>()) {
    // Don't use fallback for modelled ops.
    if (opName.isRegistered()) return nullptr;

    // Only use fallback interface for known not-stateful ops.
    const machina::OpRegistrationData *op_reg_data = nullptr;
    absl::Status s = machina::OpRegistry::Global()->LookUp(
        opName.stripDialect().str(), &op_reg_data);
    return (s.ok() && !op_reg_data->op_def.is_stateful())
               ? fallback_effect_op_interface_
               : nullptr;
  }

  return nullptr;
}

// Returns true if the op can have side effects.
bool TensorFlowDialect::CanHaveSideEffects(Operation *op) {
  // If the op has no memory side effects, it has no side effects
  if (isMemoryEffectFree(op)) return false;

  // If the op is marked stateless using the `is_stateless` attribute, then
  // it has no side effects.
  if (auto is_stateless = op->getAttrOfType<BoolAttr>("is_stateless"))
    return !is_stateless.getValue();

  // Terminators defined in the TF dialect do not have side effects.
  if (op->hasTrait<OpTrait::IsTerminator>()) return false;

  // Otherwise assume that the op can have side effects.
  return true;
}

// Hook functions which may add additional operations to the dialect.
// These are invoked at construction time.
static DenseMap<TypeID, TensorFlowDialect::AdditionalOpFunction>
    &GetAdditionalOperationHooks() {
  static auto *additional_operation_hooks =
      new DenseMap<TypeID, TensorFlowDialect::AdditionalOpFunction>();
  return *additional_operation_hooks;
}

void TensorFlowDialect::RegisterAdditionalOperationHook(
    TypeID id, AdditionalOpFunction fn) {
  GetAdditionalOperationHooks().try_emplace(id, std::move(fn));
}

TensorFlowDialect::ConstantFoldHook TensorFlowDialect::constant_fold_hook_;

TensorFlowDialect::TensorFlowDialect(MLIRContext *context)
    : Dialect(/*name=*/"tf", context, TypeID::get<TensorFlowDialect>()) {
  context->getOrLoadDialect<::mlir::tf_type::TFTypeDialect>();
  addOperations<
#define GET_OP_LIST
#include "machina/compiler/mlir/machina/ir/tf_all_ops.cc.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "machina/compiler/mlir/machina/ir/host_runtime/tfrt_ops.cc.inc"
      >();
  addInterfaces<TFInlinerInterface, TFConstantFoldInterface>();
  fallback_effect_op_interface_ =
      new TensorFlowRegistryEffectInterfaceFallback();

  // Support unknown operations because not all TensorFlow operations are
  // registered.
  allowUnknownOperations();

  for (auto &hook : GetAdditionalOperationHooks()) {
    hook.second(*this);
  }
}

TensorFlowDialect::~TensorFlowDialect() {
  delete fallback_effect_op_interface_;
}

Type TensorFlowDialect::parseType(DialectAsmParser &parser) const {
  StringRef spec = parser.getFullSymbolSpec();
  toolchain::SMLoc loc = parser.getCurrentLocation();
  parser.emitError(
      loc, "tf dialect has no types, potentially meant !tf_type." + spec);
  return nullptr;
}

Attribute TensorFlowDialect::parseAttribute(DialectAsmParser &parser,
                                            Type type) const {
  StringRef spec = parser.getFullSymbolSpec();
  toolchain::SMLoc loc = parser.getCurrentLocation();
  parser.emitError(
      loc, "tf dialect has no attributes, potentially meant #tf_type." + spec);
  return nullptr;
}

Operation *TensorFlowDialect::materializeConstant(OpBuilder &builder,
                                                  Attribute value, Type type,
                                                  Location loc) {
  return builder.create<ConstOp>(loc, type, value);
}

}  // namespace TF
}  // namespace mlir
