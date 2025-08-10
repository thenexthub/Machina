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

#ifndef MACHINA_DTENSOR_MLIR_TOPOLOGICAL_ITERATOR_H_
#define MACHINA_DTENSOR_MLIR_TOPOLOGICAL_ITERATOR_H_

#include "toolchain/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/IR/Visitors.h"  // part of Codira Toolchain

namespace machina {
namespace dtensor {

// A general Iterator that visits a FuncOp's body in topological order. Note
// that this does not visit the given FuncOp itself. Function ops are visited
// exactly once if functions are used in multiple call sites.
//
// An example usage of this Iterator is for SPMD Expansion or Sparse
// Expansion, where we expand ops in topological order starting from the
// `main` FuncOp, only visiting function ops once so that we don't expand
// multiple times.
class TopologicalIterator {
 public:
  explicit TopologicalIterator(mlir::func::FuncOp main_func);

  // Returns whether there is any further ops to visit.
  bool hasNext();

  // Returns the next op to visit in the topological ordering. Returns
  // a nullptr if there is no next op to visit.
  mlir::Operation* next();

 private:
  // Stack to keep track of ops to visit.
  toolchain::SmallVector<mlir::Operation*, 4> ops_to_visit_;

  // Keep track of functions we are walking, this is needed to avoid recursive
  // function calls.
  toolchain::SmallDenseSet<mlir::StringRef, 4> funcs_visited_in_call_stack_;

  // Keep track of all visit functions. This is to guarantee that
  // functions are visited exactly once if functions are used in multiple
  // callsites.
  toolchain::SmallDenseSet<mlir::StringRef, 4> funcs_visited_;
};

}  // namespace dtensor
}  // namespace machina

#endif  // MACHINA_DTENSOR_MLIR_TOPOLOGICAL_ITERATOR_H_
