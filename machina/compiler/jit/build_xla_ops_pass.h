/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

#ifndef MACHINA_COMPILER_JIT_BUILD_MACHINA_XLAOPS_PASS_H_
#define MACHINA_COMPILER_JIT_BUILD_MACHINA_XLAOPS_PASS_H_

#include "absl/types/optional.h"
#include "machina/core/common_runtime/optimization_registry.h"
#include "machina/core/lib/core/status.h"

namespace machina {

// Replaces TF function calls marked with `_XlaCompiledKernel` with _XlaCompile
// and _XlaRun nodes (which compile and launch, respectively, the corresponding
// HLO module).
class BuildXlaOpsPass : public GraphOptimizationPass {
 public:
  // If enable_lazy_compilation is not nullopt then *enable_lazy_compilation
  // overrides --tf_xla_enable_lazy_compilation flag in deciding whether lazy
  // compilation is enabled.
  explicit BuildXlaOpsPass(
      std::optional<bool> enable_lazy_compilation = std::nullopt)
      : enable_lazy_compilation_(enable_lazy_compilation) {}

  absl::Status Run(const GraphOptimizationPassOptions& options) override;

 private:
  std::optional<bool> enable_lazy_compilation_;
};

}  // namespace machina

#endif  // MACHINA_COMPILER_JIT_BUILD_MACHINA_XLAOPS_PASS_H_
