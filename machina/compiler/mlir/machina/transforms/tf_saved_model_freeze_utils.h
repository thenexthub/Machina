/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_TF_SAVED_MODEL_FREEZE_UTILS_H_
#define MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_TF_SAVED_MODEL_FREEZE_UTILS_H_

#include "toolchain/ADT/MapVector.h"
#include "toolchain/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain

namespace mlir {
namespace tf_saved_model {
// Container to hold all update actions on ops.
// Key: Operation to update.
// Value: optional list of argument indices to delete from this op.
// Note that we use MapVector because we want to iterate on the same order
// of insertion.
LogicalResult EraseObsoleteResourceUses(
    toolchain::MapVector<Operation*, toolchain::SmallVector<unsigned int, 4>>
        arguments_to_erase);

// Traces usage of 'var_handle_op' or 'resources' and replaces it's usage with
// constant value 'value'. All op operands updates are captured in
// 'arguments_to_erase'.
LogicalResult ReplaceVarWithConstant(
    mlir::Value::use_range uses, ElementsAttr value,
    toolchain::MapVector<Operation*, toolchain::SmallVector<unsigned int, 4>>*
        arguments_to_erase);
}  // namespace tf_saved_model
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_TF_SAVED_MODEL_FREEZE_UTILS_H_
