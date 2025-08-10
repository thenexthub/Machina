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

#include "machina/compiler/mlir/machina/utils/shape_inference_utils.h"

#include <optional>

#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/Interfaces/InferTypeOpInterface.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_dialect.h"
#include "machina/compiler/mlir/machina/utils/translate_utils.h"
#include "machina/core/ir/utils/shape_inference_utils.h"

#define DEBUG_TYPE "tf-shape-inference-utils"

namespace mlir {
namespace TF {

LogicalResult InferReturnTypeComponentsForTFOp(
    std::optional<Location> location, Operation* op, int64_t graph_version,
    tfg::OperandAsConstantFn operand_as_constant_fn,
    tfg::OpResultAsShapeFn op_result_as_shape_fn,
    tfg::ResultElementTypeFn result_element_type_fn,
    SmallVectorImpl<ShapedTypeComponents>& inferred_return_shapes) {
  assert(op->getName().getDialectNamespace() ==
         TensorFlowDialect::getDialectNamespace());
  return tfg::InferReturnTypeComponentsForTFOp(
      location, op, op->getOperands(), graph_version, operand_as_constant_fn,
      op_result_as_shape_fn, result_element_type_fn,
      machina::GetAttrValuesFromOperation, inferred_return_shapes);
}

}  // namespace TF
}  // namespace mlir
