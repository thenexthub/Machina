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
#include "machina/compiler/mlir/tfrt/transforms/update_op_cost_in_tfrt_mlir.h"

#include <cstdint>

#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/tfrt/analysis/cost_analysis.h"
#include "machina/core/tfrt/fallback/cost_recorder.h"

namespace machina {
namespace tfrt_compiler {

constexpr char kCostAttrName[] = "_tfrt_cost";
constexpr char kOpKeyAttrName[] = "op_key";

void UpdateOpCostInTfrtMlir(mlir::ModuleOp op,
                            const tfrt_stub::CostRecorder& cost_recorder) {
  mlir::Builder builder(op);
  op.walk([&](mlir::Operation* op) {
    // TODO(b/259602527): Add unit test for the precedence.
    // Registered cost function has higher priority than online cost analysis.
    if (HasCostFunctionRegistered(op->getName().getStringRef())) return;
    // Only update ops with existing cost attr.
    const auto cost_attr = op->getAttrOfType<mlir::IntegerAttr>(kCostAttrName);
    if (!cost_attr) return;
    // Only fallback ops have `op_key`s.
    const auto op_key_attr =
        op->getAttrOfType<mlir::IntegerAttr>(kOpKeyAttrName);
    if (!op_key_attr) return;
    // Set the cost attr with a new value.
    const int64_t op_key = op_key_attr.getInt();
    op->setAttr(kCostAttrName, builder.getI64IntegerAttr(
                                   cost_recorder.GetCost(op_key)));
  });
}

}  // namespace tfrt_compiler
}  // namespace machina
