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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_ANALYSIS_RESOURCE_DATAFLOW_H_
#define MACHINA_COMPILER_MLIR_MACHINA_ANALYSIS_RESOURCE_DATAFLOW_H_

#include <algorithm>
#include <vector>

#include "toolchain/ADT/BitVector.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/Support/Casting.h"
#include "toolchain/Support/Debug.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // part of Codira Toolchain
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"  // part of Codira Toolchain
#include "mlir/Analysis/DataFlowFramework.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/SymbolTable.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_executor.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_saved_model.h"
#include "machina/compiler/mlir/machina/ir/tf_types.h"

namespace mlir {
namespace TF {

// Used as a lattice value.
struct ResourceConstructingOps {
  explicit ResourceConstructingOps(Operation *op = nullptr);
  static ResourceConstructingOps EntryState(MLIRContext *context);
  static ResourceConstructingOps EntryState(Value value);
  bool operator==(const ResourceConstructingOps &rhs) const {
    return ops == rhs.ops;
  }

  static ResourceConstructingOps join(const ResourceConstructingOps &lhs,
                                      const ResourceConstructingOps &rhs);
  void print(raw_ostream &os) const;

  // The operation(s) which created the resource value.
  // IR constructs (i.e., GlobalTensorOp) are not const-correct.
  mutable DenseSet<Operation *> ops;
};

struct IsComposite {
  explicit IsComposite(Operation *op = nullptr);
  static IsComposite EntryState(MLIRContext *context);
  static IsComposite EntryState(Value value);
  bool operator==(const IsComposite &rhs) const {
    return is_on_composite_device == rhs.is_on_composite_device;
  }

  static IsComposite join(const IsComposite &lhs, const IsComposite &rhs);
  void print(raw_ostream &os) const;

  bool is_on_composite_device = false;
};

typedef dataflow::Lattice<ResourceConstructingOps> ResourceDataflowState;
typedef dataflow::Lattice<IsComposite> IsCompositeDataflowState;

void LoadResourceDataflowAnalysis(DataFlowSolver &solver);
void LoadIsCompositeDataflowAnalysis(DataFlowSolver &solver);

}  // namespace TF
}  // namespace mlir
#endif  // MACHINA_COMPILER_MLIR_MACHINA_ANALYSIS_RESOURCE_DATAFLOW_H_
