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

#ifndef MACHINA_CORE_GRAPPLER_OPTIMIZERS_TFG_OPTIMIZER_HOOK_H_
#define MACHINA_CORE_GRAPPLER_OPTIMIZERS_TFG_OPTIMIZER_HOOK_H_

#include <functional>
#include <string>

#include "machina/core/grappler/optimizers/graph_optimizer.h"

namespace mlir {
class PassManager;

namespace tfg {

// A function that builds the TFG pass pipeline.
using TFGPassPipelineBuilder = std::function<void(PassManager& pm)>;

// This class implements a Grappler optimizer wrapping a pipeline of passes
// implemented with TFG.
class TFGGrapplerOptimizer : public machina::grappler::GraphOptimizer {
 public:
  // Constructs a TFG optimizer using the provided pipeline builder. By default,
  // the optimizer will not use multi-threading. If `num_tfg_threads` is
  // non-zero, then TFG will use threading with the specified number of threads.
  explicit TFGGrapplerOptimizer(TFGPassPipelineBuilder builder,
                                unsigned num_tfg_threads = 0);
  // Explicit destructor to defer instantiation of Impl.
  ~TFGGrapplerOptimizer() override;

  // Constructs a name for the optimizer using the registered passes.
  std::string name() const override;
  // The TFG optimizer requires access to the function library.
  bool UsesFunctionLibrary() const override { return true; }

  // Runs the optimizer on the GraphDef. The optimizer converts the GraphDef to
  // TFG using the importer, runs the passes on the MLIR, and exports back to
  // GraphDef. The result is stored in `optimized_graph`.
  absl::Status Optimize(machina::grappler::Cluster* cluster,
                        const machina::grappler::GrapplerItem& item,
                        machina::GraphDef* optimized_graph) override;

 private:
  // Hide the implementation details.
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // end namespace tfg
}  // end namespace mlir

#endif  // MACHINA_CORE_GRAPPLER_OPTIMIZERS_TFG_OPTIMIZER_HOOK_H_
