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

#include "machina/core/ir/tf_op_registry.h"

#include "mlir/IR/Dialect.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/core/framework/op.h"
#include "machina/core/framework/op_def_builder.h"
#include "machina/core/ir/interfaces.h"
#include "machina/core/ir/ops.h"

namespace mlir {
namespace tfg {
TensorFlowOpRegistryInterface::TensorFlowOpRegistryInterface(Dialect *dialect)
    : TensorFlowOpRegistryInterface(dialect, machina::OpRegistry::Global()) {
}

// Returns true if the op is stateful.
static bool IsStatefulImpl(const machina::OpRegistry *registry,
                           StringRef op_name) {
  const machina::OpRegistrationData *op_reg_data =
      registry->LookUp(op_name.str());
  // If an op definition was not found, conservatively assume stateful.
  if (!op_reg_data) return true;
  return op_reg_data->op_def.is_stateful();
}

bool TensorFlowOpRegistryInterface::isStateful(Operation *op) const {
  // Handle TFG internal ops.
  if (op->hasTrait<OpTrait::IntrinsicOperation>()) return false;
  if (auto func = dyn_cast<GraphFuncOp>(op)) return func.getIsStateful();
  // Handle TFG region ops.
  // TODO(jeffniu): Region ops should be marked with a trait.
  StringRef op_name = op->getName().stripDialect();
  if (op->getNumRegions() && op_name.ends_with("Region"))
    op_name = op_name.drop_back(/*len("Region")=*/6);
  return IsStatefulImpl(registry_, op_name);
}
}  // namespace tfg
}  // namespace mlir
