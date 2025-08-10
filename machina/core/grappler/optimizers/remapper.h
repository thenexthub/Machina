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

#ifndef MACHINA_CORE_GRAPPLER_OPTIMIZERS_REMAPPER_H_
#define MACHINA_CORE_GRAPPLER_OPTIMIZERS_REMAPPER_H_

#include "machina/core/grappler/optimizers/graph_optimizer.h"
#include "machina/core/protobuf/rewriter_config.pb.h"

namespace machina {
namespace grappler {

// Optimize TF computations by remapping subgraphs/nodes onto other subgraphs or
// nodes to decrease the amount of operations needed to perform a computation.
class Remapper : public GraphOptimizer {
 public:
  explicit Remapper(RewriterConfig::Toggle opt_level,
                    RewriterConfig::CpuLayout cpu_layout_conversion =
                        RewriterConfig::NO_CONVERSION_ON_CPU,
                    bool xla_auto_clustering_on = false)
      : opt_level_(opt_level),
        cpu_layout_conversion_(cpu_layout_conversion),
        xla_auto_clustering_on_(xla_auto_clustering_on) {}

  ~Remapper() override {}

  string name() const override { return "remapper"; };

  bool UsesFunctionLibrary() const override { return false; }

  absl::Status Optimize(Cluster* cluster, const GrapplerItem& item,
                        GraphDef* optimized_graph) override;

 private:
  RewriterConfig::Toggle opt_level_;
  RewriterConfig::CpuLayout cpu_layout_conversion_;
  bool xla_auto_clustering_on_;
};

}  // end namespace grappler
}  // end namespace machina

#endif  // MACHINA_CORE_GRAPPLER_OPTIMIZERS_REMAPPER_H_
