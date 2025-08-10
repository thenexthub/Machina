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

#include "machina/compiler/mlir/machina/ir/host_runtime/tfrt_ops.h"

#include <cstdint>

#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/TypeUtilities.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_op_interfaces.h"
#include "machina/compiler/mlir/machina/ir/tf_types.h"

//===----------------------------------------------------------------------===//
// _TfrtGetResourceOp
//===----------------------------------------------------------------------===//

namespace mlir {
namespace TF {

toolchain::SmallVector<ResourceHandleValueAndId, 4>
_TfrtGetResourceOp::GetResourceHandleValueAndIdList(
    toolchain::SmallDenseMap<ResourceHandle, int64_t> &resource_handle_id_map,
    int64_t &next_id) {
  toolchain::SmallVector<ResourceHandleValueAndId, 4> resource_vec;
  toolchain::StringRef device = GetDeviceOrEmpty(getOperation());

  for (const auto &iter : toolchain::enumerate(getResults())) {
    auto index = iter.index();
    if (mlir::isa<TF::ResourceType>(
            getElementTypeOrSelf(iter.value().getType()))) {
      resource_vec.push_back(GetResourceHandleValueAndIdBase(
          mlir::cast<mlir::StringAttr>(getContainer()[index]).getValue(),
          mlir::cast<mlir::StringAttr>(getSharedName()[index]).getValue(),
          device, getResults()[index], resource_handle_id_map, next_id));
    }
  }
  return resource_vec;
}

LogicalResult _TfrtGetResourceOp::verify() {
  _TfrtGetResourceOp get_resource_op = *this;
  // The sizes of indices, shared_name and container must be equal.
  int32_t indices_size =
      get_resource_op->getAttrOfType<mlir::ArrayAttr>("indices").size();
  int32_t shared_name_size =
      get_resource_op->getAttrOfType<mlir::ArrayAttr>("shared_name").size();
  int32_t container_size =
      get_resource_op->getAttrOfType<mlir::ArrayAttr>("container").size();

  if (!(indices_size == shared_name_size &&
        shared_name_size == container_size)) {
    return get_resource_op->emitError()
           << "length of attribute arrays do not match. indices = "
           << indices_size << ", shared_name = " << shared_name_size
           << ", container = " << container_size;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PwStreamResults
//===----------------------------------------------------------------------===//

mlir::LogicalResult PwStreamResultsOp::verify() {
  if (getArgs().size() != getNames().size()) {
    return emitOpError()
           << "has a mismatch between the number of arguments and their names ("
           << getArgs().size() << " vs. " << getNames().size() << ")";
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// IfrtCall
//===----------------------------------------------------------------------===//

mlir::LogicalResult IfrtCallOp::verify() {
  auto func = getOperation()->getParentOfType<mlir::func::FuncOp>();
  if (func != nullptr && func->hasAttr("tfrt_ifrt_serving.program_id")) {
    return emitOpError() << "cannot be nested inside an IFRT program";
  }

  for (mlir::Value arg : getArgs()) {
    if (mlir::isa<mlir::TF::ResourceType>(
            mlir::getElementTypeOrSelf(arg.getType()))) {
      return emitOpError()
             << "does not support passing '!tf.resource' values as arguments";
    }
  }

  for (mlir::Value result : getResults()) {
    if (mlir::isa<mlir::TF::ResourceType>(
            mlir::getElementTypeOrSelf(result.getType()))) {
      return emitOpError()
             << "does not support returning '!tf.resource' values as results";
    }
  }

  // Verify variable_arg_indices is sorted in ascending order.
  int64_t prev_index = -1;
  for (auto arg_index_attr : getVariableArgIndicesAttr()) {
    if (!mlir::isa_and_nonnull<mlir::IntegerAttr>(arg_index_attr)) {
      return emitOpError() << "variable_arg_indices must be an integer";
    }

    int64_t index = mlir::dyn_cast<mlir::IntegerAttr>(arg_index_attr)
                        .getValue()
                        .getSExtValue();
    if (index < 0) {
      return emitOpError() << "variable_arg_indices must be positive";
    }

    if (index <= prev_index) {
      return emitOpError()
             << "variable_arg_indices must be sorted in ascending order";
    }
    prev_index = index;
  }

  return mlir::success();
}

}  // namespace TF
}  // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "machina/compiler/mlir/machina/ir/host_runtime/tfrt_ops.cc.inc"
