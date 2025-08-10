/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, August 10, 2025.
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

#ifndef MACHINA_MACHINA_XLA_STREAM_EXECUTOR_CUDA_CUDNN_SDPA_SCORE_MOD_H_
#define MACHINA_MACHINA_XLA_STREAM_EXECUTOR_CUDA_CUDNN_SDPA_SCORE_MOD_H_

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend.h"
#include "machina/xla/hlo/ir/hlo_computation.h"

namespace stream_executor {
namespace gpu {

using Tensor = std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>;
using Graph = std::shared_ptr<cudnn_frontend::graph::Graph>;
using UidGenerator = absl::FunctionRef<int64_t()>;

class ScoreModFunc {
 public:
  ScoreModFunc(const xla::HloComputation* fwd_comp,
               const xla::HloComputation* bwd_comp);

  absl::Status UpdateCudnnMap(cudnn_frontend::graph::Graph& graph,
                              UidGenerator next_uid);

  Tensor Forward(Graph graph, Tensor attention_score);

  Tensor Backward(Graph graph, Tensor grad);

  Tensor Compile(
      Graph graph,
      absl::flat_hash_map<const xla::HloInstruction*, Tensor>& hlo_to_cudnn,
      const xla::HloComputation* computation);

 private:
  absl::Status UpdateHloParameterToCudnnMap(
      cudnn_frontend::graph::Graph& graph,
      absl::flat_hash_map<const xla::HloInstruction*, Tensor>& hlo_to_cudnn,
      const xla::HloComputation* computation, UidGenerator next_uid);

  absl::Status UpdateHloConstantToCudnnMap(
      cudnn_frontend::graph::Graph& graph,
      absl::flat_hash_map<const xla::HloInstruction*, Tensor>& hlo_to_cudnn,
      const xla::HloComputation* computation);
  std::vector<Tensor> fwd_parameters_;
  const xla::HloComputation* fwd_comp_;
  const xla::HloComputation* bwd_comp_;
  absl::flat_hash_map<const xla::HloInstruction*, Tensor> fwd_hlo_to_cudnn_;
  absl::flat_hash_map<const xla::HloInstruction*, Tensor> bwd_hlo_to_cudnn_;
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // MACHINA_MACHINA_XLA_STREAM_EXECUTOR_CUDA_CUDNN_SDPA_SCORE_MOD_H_
