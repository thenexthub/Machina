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

#include "machina/compiler/mlir/machina/ir/tf_remaining_ops.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>

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
#include "mlir/Parser/Parser.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "mlir/Transforms/InliningUtils.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_attributes.h"
#include "machina/compiler/mlir/machina/ir/tf_op_interfaces.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_side_effects.h"
#include "machina/compiler/mlir/machina/ir/tf_structs.h"
#include "machina/compiler/mlir/machina/ir/tf_types.h"
#include "machina/compiler/mlir/machina/transforms/rewrite_util.h"
#include "machina/compiler/mlir/machina/utils/deserialize_mlir_module_utils.h"
#include "machina/core/platform/logging.h"
#include "machina/core/util/tensor_format.h"

namespace mlir {
namespace TF {
namespace {
#include "machina/compiler/mlir/machina/transforms/generated_canonicalize.inc"
}  // namespace

//===----------------------------------------------------------------------===//
// _XlaHostComputeOp
//===----------------------------------------------------------------------===//

// This verifies that `_XlaHostComputeMlirOp` has a well-formed
// `host_mlir_module` attribute.
// For other attributes, there is no additional verification beyond the default.
LogicalResult _XlaHostComputeMlirOp::verify() {
  _XlaHostComputeMlirOp op = *this;
  // Extract the module and function.
  StringRef host_module = op.getHostMlirModule();

  if (host_module.empty()) return success();

  mlir::OwningOpRef<mlir::ModuleOp> module_for_func;
  absl::Status status = machina::DeserializeMlirModule(
      host_module.str(), op->getContext(), &module_for_func);
  if (!status.ok()) {
    return op.emitError()
           << "attribute 'host_mlir_module' can not be deserialized. "
           << status.message();
  }

  func::FuncOp func = module_for_func->lookupSymbol<func::FuncOp>("host_func");
  if (!func)
    return op.emitError()
           << "serialized module in attribute 'host_mlir_module' does not "
              "contain 'host_func' function.";

  if (op->getNumOperands() != func.getFunctionType().getNumInputs())
    return op.emitError()
           << "'host_func' has " << func.getFunctionType().getNumInputs()
           << " inputs and '_XlaHostComputeMlir' has " << op->getNumOperands()
           << " operands.  Number of operands/inputs should be the same.";

  if (op->getNumResults() != func.getFunctionType().getNumResults())
    return op.emitError() << "'host_func' has "
                          << func.getFunctionType().getNumResults()
                          << " results and '_XlaHostComputeMlir' has "
                          << op->getNumResults()
                          << " results.  Number of results should be the same.";

  return success();
}

func::FuncOp _XlaHostComputeMlirOp::GetHostFunc(
    mlir::OwningOpRef<mlir::ModuleOp>* mlir_module) {
  if (!machina::DeserializeMlirModule(getHostMlirModule().str(),
                                         this->getContext(), mlir_module)
           .ok())
    return nullptr;
  return (*mlir_module)->lookupSymbol<func::FuncOp>("host_func");
}

//===----------------------------------------------------------------------===//
// XLA Send/Recv ops
//===----------------------------------------------------------------------===//

// For XLA Send/Recv ops the key corresponds to the resource instance.

std::optional<std::string> _XlaRecvAtHostOp::GetResourceInstanceStr() {
  return getKey().str();
}

std::optional<std::string> _XlaRecvAtHostV2Op::GetResourceInstanceStr() {
  return getKey().str();
}

std::optional<std::string> _XlaSendFromHostOp::GetResourceInstanceStr() {
  return getKey().str();
}

std::optional<std::string> _XlaSendFromHostV2Op::GetResourceInstanceStr() {
  return getKey().str();
}

namespace {
std::string GetRendezvousKey(const std::string& send_device,
                             const uint64_t send_device_incarnation,
                             const std::string& recv_device,
                             const std::string& tensor_name) {
  return absl::StrCat(send_device, ";", send_device_incarnation, ";",
                      recv_device, ";", tensor_name);
}
}  // namespace

std::optional<std::string> _HostRecvOp::GetResourceInstanceStr() {
  return GetRendezvousKey(getSendDevice().str(), getSendDeviceIncarnation(),
                          getRecvDevice().str(), getTensorName().str());
}

std::optional<std::string> _HostSendOp::GetResourceInstanceStr() {
  return GetRendezvousKey(getSendDevice().str(), getSendDeviceIncarnation(),
                          getRecvDevice().str(), getTensorName().str());
}

std::optional<std::string> _RecvOp::GetResourceInstanceStr() {
  return GetRendezvousKey(getSendDevice().str(), getSendDeviceIncarnation(),
                          getRecvDevice().str(), getTensorName().str());
}

std::optional<std::string> _SendOp::GetResourceInstanceStr() {
  return GetRendezvousKey(getSendDevice().str(), getSendDeviceIncarnation(),
                          getRecvDevice().str(), getTensorName().str());
}

}  // namespace TF
}  // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "machina/compiler/mlir/machina/ir/tf_remaining_ops.cc.inc"
