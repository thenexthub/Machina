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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_ANALYSIS_RESOURCE_VALUE_TYPED_ANALYZER_H_
#define MACHINA_COMPILER_MLIR_MACHINA_ANALYSIS_RESOURCE_VALUE_TYPED_ANALYZER_H_

#include <tuple>

#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Region.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

namespace mlir {
namespace TF {

class ResourceAnalyzer {
 public:
  explicit ResourceAnalyzer(ModuleOp module, bool skip_session_init = false);

  bool IsPotentiallyWritten(Value resource) const;

 private:
  // Analyze the specified region for resource mutating operations, namely
  // TF::AssignVariableOp, if so, set the resource associated as "potentially
  // written".
  LogicalResult AnalyzeRegion(Region& region);

  // If an op is not one of the handled ones, we assume all resource usages
  // within its purview are mutating in nature.
  void PropagatePotentiallyWrittenWithinUnhandledOp(Operation* op);

  // Given a Region associated with the callee and operands from the
  // corresponding callOp, propagate the potentially written decision to the
  // callOp's operands, if the corresponding region's arguments are potentially
  // written resources.
  void PropagatePotentiallyWrittenUpFromCallee(
      Region& region, Operation::operand_range propagate_to);

  // Marks 'resource' as written.
  void SetPotentiallyWritten(Value resource);

  struct ResourceInfo {
    bool potentially_written = false;
  };
  // Key: Resource Value's
  // Value: Information we know about that Value.
  // Note that these Value's are in general in different functions.
  DenseMap<Value, ResourceInfo> resource_infos_;
  // The set of regions we already discovered.
  DenseSet<Region*> discovered_;
  // Identifiers about mutable variables.
  // All variables are identified by (device, container, shared_name).
  DenseSet<std::tuple<toolchain::StringRef, toolchain::StringRef, toolchain::StringRef>>
      mutable_variables_;
};

}  // namespace TF
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_MACHINA_ANALYSIS_RESOURCE_VALUE_TYPED_ANALYZER_H_
