/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_COMMON_COST_H_
#define MACHINA_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_COMMON_COST_H_

#include <string>

#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain

namespace mlir {
namespace TFL {
namespace tac {

// Cost attribute string on the TFL dialect.
constexpr char kCost[] = "tac.cost";

inline void UpdateCost(Operation* op, float cost, OpBuilder* builder) {
  op->setAttr(kCost, builder->getF32FloatAttr(cost));
}

// Get the cost annotated with kCost.
inline bool GetCostOnOp(Operation* op, float* cost) {
  auto cost_type = op->getAttrOfType<FloatAttr>(kCost);
  if (cost_type == nullptr) {
    return false;
  }

  *cost = cost_type.getValueAsDouble();
  return true;
}

}  // namespace tac
}  // namespace TFL
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_COMMON_COST_H_
