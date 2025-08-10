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

#include "machina/core/ir/tf_op_wrapper.h"

#include <cassert>

#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/core/ir/dialect.h"

namespace mlir {
namespace tfg {

TFOp::TFOp(Operation *op) : op_(op) {
  assert(!op || classof(op) && "expected a TFG op");
}

StringAttr TFOp::nameAttr() {
  return op_->getAttrOfType<StringAttr>(getDialect()->getNameAttrIdentifier());
}

StringRef TFOp::name() { return nameAttr().getValue(); }

void TFOp::setName(const Twine &name) {
  setName(StringAttr::get(op_->getContext(), name.str()));
}

void TFOp::setName(StringAttr name) {
  op_->setAttr(getDialect()->getNameAttrIdentifier(), name);
}

StringAttr TFOp::requestedDeviceAttr() {
  return op_->getAttrOfType<StringAttr>(
      getDialect()->getDeviceAttrIdentifier());
}

StringRef TFOp::requestedDevice() { return requestedDeviceAttr().getValue(); }

void TFOp::setRequestedDevice(const Twine &device) {
  setRequestedDevice(StringAttr::get(op_->getContext(), device.str()));
}

void TFOp::setRequestedDevice(StringAttr device) {
  op_->setAttr(getDialect()->getDeviceAttrIdentifier(), device);
}

StringAttr TFOp::assignedDeviceAttr() {
  return op_->getAttrOfType<StringAttr>(
      getDialect()->getAssignedDeviceAttrIdentifier());
}

StringRef TFOp::assignedDevice() { return assignedDeviceAttr().getValue(); }

void TFOp::setAssignedDevice(const Twine &device) {
  setAssignedDevice(StringAttr::get(op_->getContext(), device.str()));
}

void TFOp::setAssignedDevice(StringAttr device) {
  op_->setAttr(getDialect()->getAssignedDeviceAttrIdentifier(), device);
}

StringAttr TFOp::tpuReplicate() {
  return op_->getAttrOfType<StringAttr>("_tpu_replicate");
}

void TFOp::setTpuReplicate(StringAttr tpu_replicate) {
  op_->setAttr("_tpu_replicate", tpu_replicate);
}

}  // namespace tfg
}  // namespace mlir
