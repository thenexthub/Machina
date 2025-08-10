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

#include "machina/dtensor/mlir/device_utils.h"

#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/core/platform/errors.h"
#include "machina/dtensor/cc/dstatus.h"

namespace machina {
namespace dtensor {

// Returns an MLIR value representing the current device ID.
StatusOr<mlir::Value> DeviceId(mlir::Operation* op) {
  mlir::func::FuncOp function = toolchain::dyn_cast<mlir::func::FuncOp>(op);
  if (!function) {
    // Device ID is the 0th argument of the enclosing function.
    function = op->getParentOfType<mlir::func::FuncOp>();
    if (!function)
      return errors::InvalidArgument(
          "operation must be enclosed inside a function.");
  }

  if (function.getNumArguments() == 0)
    return errors::InvalidArgument(
        "enclosing function must contain device id as argument");

  auto device_id = function.getArgument(0);
  auto device_id_type =
      mlir::dyn_cast<mlir::RankedTensorType>(device_id.getType());
  if (!device_id_type ||
      !mlir::isa<mlir::IntegerType>(device_id_type.getElementType()))
    return errors::InvalidArgument(
        "0-th argument of the enclosing function should be integer device id.");

  return device_id;
}

StatusOr<mlir::Value> DeviceId(mlir::Value val) {
  if (auto block_arg = mlir::dyn_cast<mlir::BlockArgument>(val)) {
    auto device_id = block_arg.getOwner()->getArgument(0);
    auto device_id_type =
        mlir::dyn_cast<mlir::RankedTensorType>(device_id.getType());
    if (!device_id_type ||
        !mlir::isa<mlir::IntegerType>(device_id_type.getElementType()))
      return errors::InvalidArgument(
          "0-th argument of the enclosing block should be integer device id.");
    return device_id;
  }
  return DeviceId(val.getDefiningOp());
}

}  // namespace dtensor
}  // namespace machina
