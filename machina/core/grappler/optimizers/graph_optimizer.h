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

#ifndef MACHINA_CORE_GRAPPLER_OPTIMIZERS_GRAPH_OPTIMIZER_H_
#define MACHINA_CORE_GRAPPLER_OPTIMIZERS_GRAPH_OPTIMIZER_H_

#include <string>

#include "absl/status/status.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace grappler {

class Cluster;
struct GrapplerItem;

// An abstract interface for an algorithm for generating a candidate
// optimization of a GrapplerItem for running on a cluster.
class GraphOptimizer {
 public:
  GraphOptimizer() : deadline_usec_(0) {}
  virtual ~GraphOptimizer() {}

  virtual string name() const = 0;

  // Returns true if the optimizer requires a valid function library to perform
  // graph optimization. If false, optimized GrapplerItem will have a stub
  // instead of real function library (all function signatures and attributes
  // will be valid, but function body will be empty). Most of the optimizers
  // that do not instantiate functions should return true.
  virtual bool UsesFunctionLibrary() const = 0;

  // Routine called to allow an algorithm to propose a rewritten graph
  // for the graph, feeds and fetches in "item" to run more efficiently
  // on "cluster". If the returned status is OkStatus() then
  // *optimized_graph contains the rewritten graph.
  // Returns an error status if it failed to generate a solution.
  //
  // A return value of error::Aborted() can be used signal early termination of
  // the optimizer, e.g. if the optimization turned out to be a no-op. In this
  // case the content of *optimized_graph is undefined.
  virtual absl::Status Optimize(Cluster* cluster, const GrapplerItem& item,
                                GraphDef* optimized_graph) = 0;

  // Subclasses may define a version of Optimize that consumes item.
  virtual absl::Status Optimize(Cluster* cluster, GrapplerItem&& item,
                                GraphDef* optimized_graph) {
    return Optimize(cluster, item, optimized_graph);
  }

  // Set deadline in microseconds since epoch. A value of zero means no
  // deadline.
  void set_deadline_usec(uint64 deadline_usec) {
    deadline_usec_ = deadline_usec;
  }
  uint64 deadline_usec() const { return deadline_usec_; }
  bool DeadlineExceeded() const {
    return deadline_usec_ > 0 && Env::Default()->NowMicros() > deadline_usec_;
  }

 private:
  uint64 deadline_usec_;
};

#define GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED()                \
  do {                                                        \
    if (this->DeadlineExceeded()) {                           \
      return absl::DeadlineExceededError(                     \
          absl::StrCat(this->name(), " exceeded deadline.")); \
    }                                                         \
  } while (0)

}  // end namespace grappler
}  // end namespace machina

#endif  // MACHINA_CORE_GRAPPLER_OPTIMIZERS_GRAPH_OPTIMIZER_H_
