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

#include "machina/compiler/mlir/lite/utils/stateful_ops_utils.h"

#include <vector>

#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"
#include "machina/compiler/mlir/lite/utils/utils.h"

namespace mlir {
namespace TFL {

bool IsStatefulOp(Operation* op, std::vector<int>* stateful_operand_indices) {
  if (auto stateful_op = dyn_cast_or_null<StatefulOpInterface>(op)) {
    *stateful_operand_indices = stateful_op.GetStatefulOperands();
    return true;
  }

  return false;
}

}  // namespace TFL
}  // namespace mlir
