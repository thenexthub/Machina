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

#include "machina/dtensor/mlir/utils/dtensor_mlir_passes_internal.h"

#include <cstdlib>
#include <cstring>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "machina/dtensor/mlir/create_dtensor_mlir_passes.h"


namespace machina {
namespace dtensor {

// Combine independent DTensorAllReduceOps from the same ClusterOp.
// Non-sea of donuts does not need this. It can rely on the XLA all-reduce
// combiner instead.
void AddDTensorAllReduceCombineOptimization(mlir::OpPassManager* pm) {
  // Experimental feature. If zero, the optimization for combining all reduces
  // with same group assignment and reduction, will not be done.
  const char* env_str =
      (std::getenv("DTENSOR_ENABLE_COMBINE_ALL_REDUCES_OPTIMIZATION"));
  if (env_str && strcmp(env_str, "0") == 0) {
    return;
  }
  pm->addNestedPass<mlir::func::FuncOp>(
      CreateDTensorAllReduceCombineOptimization());
}

}  // namespace dtensor
}  // namespace machina

