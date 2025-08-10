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

#ifndef MACHINA_MACHINA_XLA_SERVICE_GPU_TRANSFORMS_WINDOWED_EINSUM_HANDLER_H_
#define MACHINA_MACHINA_XLA_SERVICE_GPU_TRANSFORMS_WINDOWED_EINSUM_HANDLER_H_

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "machina/xla/hlo/ir/hlo_computation.h"
#include "machina/xla/hlo/ir/hlo_instruction.h"
#include "machina/xla/hlo/ir/hlo_module.h"
#include "machina/xla/hlo/pass/hlo_pass_interface.h"

namespace xla::gpu {

// This pass is targeting the windowed einsum optimization
// in the SPMD pipeline. The SPMD partitioner's HandleDot
// rewrites all-gather+gemm or gemm+reduce-scatter into sharded
// loops with p2p communication to achieve overlap between sharded
// gemms and communication.
// This pass does the following transformations to further optimize
// rewritten loops for GPU:
//  1. it annotates independent gemms with stream ids in the backend config
//     to achieve overlap between compute kernels in addition to
//     compute-gemm overlap
//  2. it removes all-gathers in ag+gemm patterns if the input to the ags is
//     also an input to another windowed einsum ag loop, ie:
//                       input
//                       /    |
//                      /     |
//                     AG    windowed loop
//                     /
//                    /
//                   dot
// to:
//                       input
//                       |
//                       |
//                     windowed loop
//                       |
//                       |
//                      dot
//  3. it moves the fp8 dequantization outside of a rewritten windowed einsum
//     loop inside of the loop so the dq+gemm can be fused into an fp8 gemm
//     later in gemm rewriter.
//  4. it shards a large all-to-all+gemm into smaller independent a2a+gemm
//     shards to pipeline compute and communication so most of the
//     communication overhead can be hidden.

class WindowedEinsumHandler : public HloModulePass {
 public:
  absl::string_view name() const override { return "windowed-einsum-handler"; }

  struct WindowedEinsumAgLoops {
    explicit WindowedEinsumAgLoops(HloInstruction* loop) : loop(loop) {}
    HloInstruction* loop;
    bool consumed = false;
  };

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  constexpr static const char* kWindowedEinsumRsLoopName =
      "windowed_dot_general_body_rs";
  constexpr static const char* kWindowedEinsumAgLoopName =
      "windowed_dot_general_body_ag";

 private:
  std::vector<WindowedEinsumAgLoops> all_ag_loops_;
};

}  // namespace xla::gpu

#endif  // MACHINA_MACHINA_XLA_SERVICE_GPU_TRANSFORMS_WINDOWED_EINSUM_HANDLER_H_
