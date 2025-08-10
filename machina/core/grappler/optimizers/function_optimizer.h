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

#ifndef MACHINA_CORE_GRAPPLER_OPTIMIZERS_FUNCTION_OPTIMIZER_H_
#define MACHINA_CORE_GRAPPLER_OPTIMIZERS_FUNCTION_OPTIMIZER_H_

#include "machina/core/grappler/optimizers/graph_optimizer.h"
#include "machina/core/protobuf/rewriter_config.pb.h"

namespace machina {
namespace grappler {

// Remap TensorFlow subgraphs onto alternative operations or collection of
// operations to make the overall graph more efficient.
class FunctionOptimizer : public GraphOptimizer {
 public:
  explicit FunctionOptimizer(RewriterConfig::Toggle opt_level,
                             bool lower_control_flow)
      : opt_level_(opt_level), lower_control_flow_(lower_control_flow) {}
  ~FunctionOptimizer() override = default;

  string name() const override { return "function_optimizer"; };

  bool UsesFunctionLibrary() const override { return true; }

  absl::Status Optimize(Cluster* cluster, const GrapplerItem& item,
                        GraphDef* optimized_graph) override;

 private:
  friend class FunctionOptimizerTest;

  // Runs a single function optimizer pass over the `graph`. All nodes that are
  // not function calls will be copied from the `graph` to the
  // `optimized_graph`. Function call nodes inlined or specialized, and
  // instantiated function body or specialized function call nodes will be added
  // to the `optimized_graph`.
  absl::Status RunFunctionOptimizerPass(const GrapplerItem& item,
                                        GraphDef* optimized_graph) const;

  RewriterConfig::Toggle opt_level_;
  bool lower_control_flow_;
};

}  // end namespace grappler
}  // end namespace machina

#endif  // MACHINA_CORE_GRAPPLER_OPTIMIZERS_FUNCTION_OPTIMIZER_H_
