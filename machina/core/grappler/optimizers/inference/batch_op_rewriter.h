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

#ifndef MACHINA_CORE_GRAPPLER_OPTIMIZERS_INFERENCE_BATCH_OP_REWRITER_H_
#define MACHINA_CORE_GRAPPLER_OPTIMIZERS_INFERENCE_BATCH_OP_REWRITER_H_

#include "machina/core/grappler/grappler_item.h"
#include "machina/core/grappler/optimizers/custom_graph_optimizer.h"
#include "machina/core/grappler/optimizers/inference/batch_op_rewriter.pb.h"

namespace machina {
namespace grappler {

constexpr char kEnableAdaptiveSchedulerAttr[] = "_enable_adaptive_scheduler";
constexpr char kMinInflightBatchesAttr[] = "_min_inflight_batches";
constexpr char kInitialInflightBatchesAttr[] = "_initial_inflight_batches";
constexpr char kMaxInflightBatchesAttr[] = "_max_inflight_batches";
constexpr char kBatchesToAverageOverAttr[] = "_batches_to_average_over";
constexpr char kFullBatchSchedulingBoostMicros[] =
    "_full_batch_scheduling_boost_micros";  // NOLINT(whitespace/line_length)

constexpr int64_t kMinInflightBatches = 16;
constexpr int64_t kInitialInflightBatches = 16;
constexpr int64_t kBatchesToAverageOver = 10;
constexpr int64_t kMaxInflightBatches = 64;

using ::machina::serving::BatchOpRewriteConfig;

// This optimization does the following:
//
// Rewrite `num_batch_threads` to zero in batch-op. In this way, graphs with
// batch op will use a shared thread pool to schedule batches, as opposed to
// allocating batch threads per batch-op.
class BatchOpRewriter : public ::machina::grappler::CustomGraphOptimizer {
 public:
  absl::Status Init(
      const ::machina::RewriterConfig_CustomGraphOptimizer* config) override;

  std::string name() const override { return "batch_op_rewriter"; }

  bool UsesFunctionLibrary() const override { return false; }

  absl::Status Optimize(::machina::grappler::Cluster* cluster,
                        const ::machina::grappler::GrapplerItem& item,
                        ::machina::GraphDef* optimized_graph) override;

 private:
  BatchOpRewriteConfig config_;
};

}  // namespace grappler
}  // namespace machina

#endif  // MACHINA_CORE_GRAPPLER_OPTIMIZERS_INFERENCE_BATCH_OP_REWRITER_H_
