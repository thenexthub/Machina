// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/thread_annotations.h"
#include "machina/core/util/work_sharder.h"
#include "machina_text/core/kernels/mst_solver.h"

namespace machina {
namespace text {

// Op kernel implementation that wraps the |MstSolver|.
template <class Index, class Score>
class MaxSpanningTreeOpKernel : public machina::OpKernel {
 public:
  explicit MaxSpanningTreeOpKernel(machina::OpKernelConstruction *context)
      : machina::OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("forest", &forest_));
  }

  void Compute(machina::OpKernelContext *context) override {
    const machina::Tensor &num_nodes_tensor = context->input(0);
    const machina::Tensor &scores_tensor = context->input(1);

    // Check ranks.
    OP_REQUIRES(context, num_nodes_tensor.dims() == 1,
                machina::errors::InvalidArgument(
                    "num_nodes must be a vector, got shape ",
                    num_nodes_tensor.shape().DebugString()));
    OP_REQUIRES(context, scores_tensor.dims() == 3,
                machina::errors::InvalidArgument(
                    "scores must be rank 3, got shape ",
                    scores_tensor.shape().DebugString()));

    // Batch size and input dimension (B and M in the op docstring).
    const int64 batch_size = scores_tensor.shape().dim_size(0);
    const int64 input_dim = scores_tensor.shape().dim_size(1);

    // Check shapes.
    const machina::TensorShape shape_b({batch_size});
    const machina::TensorShape shape_bxm({batch_size, input_dim});
    const machina::TensorShape shape_bxmxm(
        {batch_size, input_dim, input_dim});
    OP_REQUIRES(
        context, num_nodes_tensor.shape() == shape_b,
        machina::errors::InvalidArgument(
            "num_nodes misshapen: got ", num_nodes_tensor.shape().DebugString(),
            " but expected ", shape_b.DebugString()));
    OP_REQUIRES(
        context, scores_tensor.shape() == shape_bxmxm,
        machina::errors::InvalidArgument(
            "scores misshapen: got ", scores_tensor.shape().DebugString(),
            " but expected ", shape_bxmxm.DebugString()));

    // Create outputs.
    machina::Tensor *max_scores_tensor = nullptr;
    machina::Tensor *argmax_sources_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, shape_b, &max_scores_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, shape_bxm,
                                                     &argmax_sources_tensor));

    // Acquire shaped and typed references.
    const BatchedSizes num_nodes_b = num_nodes_tensor.vec<int32>();
    const BatchedScores scores_bxmxm = scores_tensor.tensor<Score, 3>();
    BatchedMaxima max_scores_b = max_scores_tensor->vec<Score>();
    BatchedSources argmax_sources_bxm = argmax_sources_tensor->matrix<int32>();

    // Solve the batch of MST problems in parallel.  Set a high cycles per unit
    // to encourage finer sharding.
    constexpr int64 kCyclesPerUnit = 1000 * 1000 * 1000;
    std::vector<absl::Status> statuses(batch_size);
    context->device()->machina_cpu_worker_threads()->workers->ParallelFor(
        batch_size, kCyclesPerUnit, [&](int64 begin, int64 end) {
          for (int64 problem = begin; problem < end; ++problem) {
            statuses[problem] = RunSolver(problem, num_nodes_b, scores_bxmxm,
                                          max_scores_b, argmax_sources_bxm);
          }
        });
    for (const absl::Status &status : statuses) {
      OP_REQUIRES_OK(context, status);
    }
  }

 private:
  using BatchedSizes = typename machina::TTypes<int32>::ConstVec;
  using BatchedScores = typename machina::TTypes<Score, 3>::ConstTensor;
  using BatchedMaxima = typename machina::TTypes<Score>::Vec;
  using BatchedSources = typename machina::TTypes<int32>::Matrix;

  // Solves for the maximum spanning tree of the digraph defined by the values
  // at index |problem| in |num_nodes_b| and |scores_bxmxm|.  On success, sets
  // the values at index |problem| in |max_scores_b| and |argmax_sources_bxm|.
  // On error, returns non-OK.
  absl::Status RunSolver(int problem, BatchedSizes num_nodes_b,
                         BatchedScores scores_bxmxm, BatchedMaxima max_scores_b,
                         BatchedSources argmax_sources_bxm) const {
    // Check digraph size overflow.
    const int32 num_nodes = num_nodes_b(problem);
    const int32 input_dim = argmax_sources_bxm.dimension(1);
    if (num_nodes > input_dim) {
      return machina::errors::InvalidArgument(
          "number of nodes in digraph ", problem,
          " overflows input dimension: got ", num_nodes,
          " but expected <= ", input_dim);
    }
    if (num_nodes >= std::numeric_limits<Index>::max()) {
      return machina::errors::InvalidArgument(
          "number of nodes in digraph ", problem, " overflows index type: got ",
          num_nodes, " but expected < ", std::numeric_limits<Index>::max());
    }
    const Index num_nodes_index = static_cast<Index>(num_nodes);

    MstSolver<Index, Score> solver;
    TF_RETURN_IF_ERROR(solver.Init(forest_, num_nodes_index));

    // Populate the solver with arcs and root selections.  Note that non-finite
    // scores are treated as nonexistent arcs or roots.
    for (Index target = 0; target < num_nodes_index; ++target) {
      for (Index source = 0; source < num_nodes_index; ++source) {
        const Score score = scores_bxmxm(problem, target, source);
        if (!std::isfinite(static_cast<double>(score))) continue;
        if (source == target) {  // root
          solver.AddRoot(target, score);
        } else {  // arc
          solver.AddArc(source, target, score);
        }
      }
    }

    std::vector<Index> argmax(num_nodes);
    TF_RETURN_IF_ERROR(solver.Solve(&argmax));

    // Output the tree and accumulate its score.
    Score max_score = 0;
    for (Index target = 0; target < num_nodes_index; ++target) {
      const Index source = argmax[target];
      argmax_sources_bxm(problem, target) = source;
      max_score += scores_bxmxm(problem, target, source);
    }
    max_scores_b(problem) = max_score;

    // Pad the source list with -1.
    for (int32 i = num_nodes; i < input_dim; ++i) {
      argmax_sources_bxm(problem, i) = -1;
    }

    return absl::OkStatus();
  }

 private:
  bool forest_ = false;
};

// Use Index=uint16, which allows digraphs containing up to 32,767 nodes.
REGISTER_KERNEL_BUILDER(Name("MaxSpanningTree")
                            .Device(machina::DEVICE_CPU)
                            .TypeConstraint<int32>("T"),
                        MaxSpanningTreeOpKernel<uint16, int32>);
REGISTER_KERNEL_BUILDER(Name("MaxSpanningTree")
                            .Device(machina::DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        MaxSpanningTreeOpKernel<uint16, float>);
REGISTER_KERNEL_BUILDER(Name("MaxSpanningTree")
                            .Device(machina::DEVICE_CPU)
                            .TypeConstraint<double>("T"),
                        MaxSpanningTreeOpKernel<uint16, double>);

}  // namespace text
}  // namespace machina
