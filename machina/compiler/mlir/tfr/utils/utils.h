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

#ifndef MACHINA_COMPILER_MLIR_TFR_UTILS_UTILS_H_
#define MACHINA_COMPILER_MLIR_TFR_UTILS_UTILS_H_

#include <string>

#include "mlir/IR/Block.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/tfr/ir/tfr_ops.h"

namespace mlir {
namespace TFR {

// This is a hardcoded rule for mapping a TF op name to the corresponding
// TFR function name. Examples:
//   tf.Pack => tf__pack
//   tf.ConcatV2 => tf__concat_v2
// TODO(fengliuai): move to an util file.
std::string GetComposeFuncName(StringRef tf_op_name);

// This is a hardcoded rule for mapping a TFR function op name to the
// corresponding TF opname. Examples:
//   tf__pack -> tf.Pack
//   tf__concat_v2 => tf.ConcatV2
std::string GetTFOpName(StringRef compose_func_name);

// Validate the attributes of 'src' is either contained in the registered
// attribute sets or in the allowed list.
LogicalResult ValidateAttrs(Operation* src, const StringSet<>& registered);

// Copies all the allowed attributes in 'src' to 'dst'. The copy failed if the
// 'dst' has the attribute. Return a failure if there are any attributes are not
// allowed and also unregistered.
LogicalResult CopyAllowedUnregisteredAttrs(Operation* src, CallOp dst,
                                           const StringSet<>& registered);

// Copies all the allowed attributes in 'src' to 'dst'. FlatSymbolRefAttr is
// excluded.
LogicalResult CopyNonSymbolRefAttrs(CallOp src, Operation* dst);

// Propagates all the attributes in 'src' to the operations between 'begin' and
// 'end'. Operation 'end' is excluded.
void PropagateAttrsToOperations(CallOp src, Block::iterator begin,
                                Block::iterator end);

}  // namespace TFR
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_TFR_UTILS_UTILS_H_
