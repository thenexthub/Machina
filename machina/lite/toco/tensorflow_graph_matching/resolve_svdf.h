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
#ifndef MACHINA_LITE_TOCO_MACHINA_GRAPH_MATCHING_RESOLVE_SVDF_H_
#define MACHINA_LITE_TOCO_MACHINA_GRAPH_MATCHING_RESOLVE_SVDF_H_

#include <memory>
#include <string>
#include <vector>

#include "machina/core/framework/attr_value.pb.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/lite/toco/model.h"
#include "machina/lite/toco/machina_graph_matching/cluster.h"
#include "machina/lite/toco/machina_graph_matching/cluster_utils.h"
#include "machina/lite/toco/tooling_util.h"

namespace toco {

class SvdfCluster : public Cluster {
 public:
  // For this cluster, it collapses all the nodes in nodes_ into a composite op
  // and it returns all the newly generated ops in new_nodes_.
  void CreateNodes() override;

  // A helper function to set the pattern of Const nodes which CreateNodes()
  // should handle specially.
  void AddConstNodePattern(const std::string& const_pattern) {
    const_node_patterns_.push_back(const_pattern);
  }

  ~SvdfCluster() override {}

 private:
  // The main function which is used to create Const nodes for this cluster.
  // These Const nodes are the inputs to the composite op generated for this
  // cluster.
  void CreateConstNode(const std::string& const_pattern);

  // Receives a vector of Const nodes, merge them (if necessary) and returns
  // only one Const node holding all the arrays contents. It transposes it if
  // needed.
  void MaybeMergeConstNodes(
      const std::vector<const machina::NodeDef*>& const_node_parts,
      bool transpose_tensor_value,
      const std::unique_ptr<machina::NodeDef>& merged_node);

  // Infer the value of Svdf filter rank, by looking up a reshape operator which
  // is used for 'output' which reshapes output from [num_filters, batch, 1]
  // shape to [num_units, rank, batch] shape. The 2nd shape element is rank.
  int InferFilterRank();

  std::vector<std::string> const_node_patterns_;
};

class SvdfClusterFactory : public ClusterFactoryInterface {
 public:
  // Creates a cluster of nodes using a name-based pattern matching approach. It
  // uses a node as a seed and if its name matches a certain pattern, then it
  // builds the cluster around that node.
  // This factory expects nodes which have "SVDF_weights_feature" and
  // "SVDF_weights_time" pattern in their names (and optionally "SVDF_bias")
  // and it creates an SVDF Op from them.
  std::unique_ptr<Cluster> CreateCluster(
      const machina::NodeDef& node,
      const machina::GraphDef& graph_def) const override;
};

}  // end namespace toco

#endif  // MACHINA_LITE_TOCO_MACHINA_GRAPH_MATCHING_RESOLVE_SVDF_H_
