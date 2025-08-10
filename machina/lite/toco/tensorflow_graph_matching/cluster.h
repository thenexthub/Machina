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
#ifndef MACHINA_LITE_TOCO_MACHINA_GRAPH_MATCHING_CLUSTER_H_
#define MACHINA_LITE_TOCO_MACHINA_GRAPH_MATCHING_CLUSTER_H_

#include <memory>
#include <string>
#include <vector>

#include "machina/core/framework/attr_value.pb.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/framework/tensor.pb.h"
#include "machina/core/framework/tensor_shape.pb.h"
#include "machina/lite/toco/model.h"
#include "machina/lite/toco/machina_graph_matching/cluster_utils.h"
#include "machina/lite/toco/tooling_util.h"

namespace toco {

// The base class for Cluster. A cluster is group of nodes all related to each
// other because their name match a given "pattern", which shows they all belong
// to a composite op supported in TFLite. The nodes in a cluster will be
// collapsed into a single composite op node plus a series of constant nodes
// holding the input parameters to that node. The nodes in a cluster are assumed
// to be using the same device. By changing the "pattern" we can have different
// subclasses of the base Cluster class.
class Cluster {
 public:
  virtual ~Cluster() {}

  virtual void CreateNodes() = 0;

  // Save the following info from the original GraphDef this cluster is from:
  // 1- a pointer to the GraphDef
  // 2- All the nodes in GraphDef which belong to this cluster.
  void SetGraphDefInfo(const machina::GraphDef* graph_def);

  const std::string& GetName() const { return name_; }

  const std::vector<std::unique_ptr<machina::NodeDef>>& GetNewNodes() const {
    return new_nodes_;
  }

  const std::vector<const machina::NodeDef*>& GetNodes() { return nodes_; }

  void SetName(const std::string& name) { name_ = name; }

  void SetDevice(const std::string& device) { device_ = device; }

  // Find the input(s) and output(s) of this Cluster.
  bool FindClusterInputsAndOutputs();

 protected:
  std::string name_;
  std::string device_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;

  // Used to hold the pointers to nodes which are in this cluster. These nodes
  // are pointing to the nodes in graph_def_.
  std::vector<const machina::NodeDef*> nodes_;

  // Used to cache the newly generated nodes: like the nodes created by
  // collapsing Const nodes, or the nodes which is used to show the composite
  // op.
  std::vector<std::unique_ptr<machina::NodeDef>> new_nodes_;

  const machina::GraphDef* graph_def_; /*Not owned*/
};

// A factory interface for cluster class.
// It defines a virtual function interface which is responsible for creating
// a cluster. Each cluster factory is responsible to pack a cluster of nodes
// into a cluster using a name-based pattern matching approach.
class ClusterFactoryInterface {
 public:
  virtual ~ClusterFactoryInterface() {}

  // Creates a cluster of nodes using a name-based pattern matching approach. It
  // uses a node as a seed and if its name matches a certain pattern, then it
  // builds the cluster around that node.
  virtual std::unique_ptr<Cluster> CreateCluster(
      const machina::NodeDef& node,
      const machina::GraphDef& graph_def) const = 0;
};

}  // end namespace toco

#endif  // MACHINA_LITE_TOCO_MACHINA_GRAPH_MATCHING_CLUSTER_H_
