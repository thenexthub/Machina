/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#ifndef MACHINA_COMPILER_JIT_INCREASE_DYNAMISM_FOR_AUTO_JIT_PASS_H_
#define MACHINA_COMPILER_JIT_INCREASE_DYNAMISM_FOR_AUTO_JIT_PASS_H_

#include "absl/status/status.h"
#include "machina/core/common_runtime/optimization_registry.h"
#include "machina/core/platform/status.h"

namespace machina {

// Increases the amount of "dynamism" representable by XLA clusters by rewriting
// the TensorFlow graph.  This pass does the following rewrites:
//
// Slice
// -----
//
//   Slice(op, begin, size <must be constant>) =>
//     Slice(op, begin, actual_size(op.shape(), size, begin));
//       _XlaCompileTimeConstantInputs={2}
//
// where
//
//   actual_size(op_shape, size, begin)[i] =
//     size[i] == -1 ? (op_shape[i] - size[i])
//                   : size[i]
//
// This pass, combined with jit/partially_decluster_pass, reduces the number of
// unnecessary cluster recompilations in some common cases.  After the rewrite
// shown above jit/partially_decluster_pass extracts the actual_size(...)
// computation to outside the XLA cluster, causing the cluster to be versioned
// only on the actual size of the XlaDynamicSlice.  This avoids recompilation
// due to superficial changes that don't affect tensor shapes.
//
// Future Work TODO(b/111210515)
// -----------------------------
//
// In the future we will also translate StridedSlice and Pad a similar way.
class IncreaseDynamismForAutoJitPass : public GraphOptimizationPass {
 public:
  absl::Status Run(const GraphOptimizationPassOptions& options) override;
};

}  // namespace machina

#endif  // MACHINA_COMPILER_JIT_INCREASE_DYNAMISM_FOR_AUTO_JIT_PASS_H_
