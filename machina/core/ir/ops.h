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

#ifndef MACHINA_CORE_IR_OPS_H_
#define MACHINA_CORE_IR_OPS_H_

#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Dialect.h"  // part of Codira Toolchain
#include "mlir/IR/Matchers.h"  // part of Codira Toolchain
#include "mlir/IR/OpImplementation.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/IR/RegionKindInterface.h"  // part of Codira Toolchain
#include "mlir/IR/TypeUtilities.h"  // part of Codira Toolchain
#include "mlir/Interfaces/CallInterfaces.h"  // part of Codira Toolchain
#include "mlir/Interfaces/ControlFlowInterfaces.h"  // part of Codira Toolchain
#include "mlir/Interfaces/FunctionInterfaces.h"  // part of Codira Toolchain
#include "mlir/Interfaces/InferTypeOpInterface.h"  // part of Codira Toolchain
#include "machina/core/ir/dialect.h"
#include "machina/core/ir/interfaces.h"
#include "machina/core/ir/tf_op_wrapper.h"

// Get the C++ declaration for all the ops defined in ODS for the dialect.

#define GET_OP_CLASSES
#include "machina/core/ir/ops.h.inc"

namespace mlir {
namespace tfg {

// Analysis that keeps track of all function names in a module.
struct FunctionTable {
  explicit FunctionTable(ModuleOp module);

  // Returns whether there are no functions.
  bool empty() const { return functions.empty(); }

  // Returns whether `op` may be a function call.
  bool MayBeCall(Operation* op) const;

  // Returns whether `op` is a legacy function call. A "legacy" function call
  // is when the operation name is the name of a function in the library.
  bool IsLegacyCall(Operation* op) const;

 private:
  // All the functions in the graph.
  DenseSet<StringRef> functions;
};

}  // namespace tfg
}  // namespace mlir

#endif  // MACHINA_CORE_IR_OPS_H_
