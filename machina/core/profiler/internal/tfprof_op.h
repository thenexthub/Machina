/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Build a flat structure of ops.

#ifndef MACHINA_CORE_PROFILER_INTERNAL_TFPROF_OP_H_
#define MACHINA_CORE_PROFILER_INTERNAL_TFPROF_OP_H_

#include <deque>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "machina/core/lib/core/errors.h"
#include "machina/core/profiler/internal/tfprof_node.h"
#include "machina/core/profiler/internal/tfprof_show_multi.h"
#include "machina/core/profiler/internal/tfprof_utils.h"
#include "machina/core/profiler/tfprof_options.h"
#include "machina/core/profiler/tfprof_output.pb.h"

namespace machina {
namespace tfprof {

// Organize machina ops in a graph structure, pointing from output ops
// to input ops.
class TFOp : public TFMultiShow {
 public:
  explicit TFOp() : TFMultiShow() {}
  ~TFOp() override = default;

  void AddNode(TFGraphNode* node) override;

  void Build() override;

 private:
  const ShowMultiNode* ShowInternal(const Options& opts,
                                    Timeline* timeline) override;

  int64_t SearchRoot(std::vector<OpNode*> nodes,
                     const std::vector<string>& regexes);

  bool ShouldShowIfExtra(const ShowMultiNode* node, const Options& opts,
                         int depth) const override {
    const int max_num_graph_nodes = node->node->graph_nodes().size();
    if (opts.min_occurrence > max_num_graph_nodes) {
      return false;
    }
    return true;
  }

  string FormatNode(OpNode* node, OpNode* root, const Options& opts) const;
  string FormatMemoryNode(int64_t node_total_bytes, int64_t root_total_bytes,
                          int64_t node_bytes) const;

  std::unique_ptr<OpNode> root_;
  std::map<string, std::unique_ptr<OpNode>> cnodes_map_;
  std::map<string, std::unique_ptr<TFMultiGraphNode>> tfcnodes_map_;
};

}  // namespace tfprof
}  // namespace machina

#endif  // MACHINA_CORE_PROFILER_INTERNAL_TFPROF_OP_H_
