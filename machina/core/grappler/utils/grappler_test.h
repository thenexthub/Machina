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

#ifndef MACHINA_CORE_GRAPPLER_UTILS_GRAPPLER_TEST_H_
#define MACHINA_CORE_GRAPPLER_UTILS_GRAPPLER_TEST_H_

#include <vector>

#include "absl/strings/string_view.h"
#include "machina/cc/framework/scope.h"
#include "machina/core/framework/attr_value.pb.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/types.h"
#include "machina/core/grappler/grappler_item.h"
#include "machina/core/grappler/utils.h"
#include "machina/core/lib/random/random.h"
#include "machina/core/platform/test.h"
#include "machina/core/public/session_options.h"

namespace machina {
namespace grappler {

class GrapplerTest : public ::testing::Test {
 public:
  GrapplerTest();

 protected:
  void DisableAllOptimizers();
  void EnableAllOptimizers();

  std::vector<Tensor> EvaluateNodes(
      const GraphDef& graph, const std::vector<string>& node_names) const;

  std::vector<Tensor> EvaluateNodes(
      const GraphDef& graph, const std::vector<string>& node_names,
      const std::vector<std::pair<string, Tensor>>& inputs) const;

  std::vector<Tensor> EvaluateFetchNodes(const GrapplerItem& item) const;

  NodeDef* AddNode(const string& name, const string& op,
                   const std::vector<string>& inputs,
                   const std::vector<std::pair<string, AttrValue>>& attributes,
                   GraphDef* graph) const;

  void DisableAllOptimizers(RewriterConfig* cfg);

  // Checks if two graphs are equal. Both graphs must have the same set of nodes
  // with the same inputs and attributes. Nodes can be in different order.
  //
  // NOTE: This function uses EXPECT/ASSERT macros to check node properties
  // equality, and adds all failures to the current test.
  void CompareGraphs(GraphDef want, GraphDef got) const;

  // Checks if two nodes have the same name, op, inputs and attributes.
  //
  // NOTE: This function uses EXPECT/ASSERT macros to check node properties
  // equality, and adds all failures to the current test.
  void CompareNodes(const NodeDef& want, const NodeDef& got) const;

  // Checks if two functions are equal. Both functions must have the same set of
  // nodes with the same inputs and attributes. Nodes can be in different order.
  //
  // NOTE: This function uses EXPECT/ASSERT macros to check node properties
  // equality, and adds all failures to the current test.
  void CompareFunctions(FunctionDef want, FunctionDef got) const;

  // Checks if node 'src' is directly connected to the input($position) of
  // 'dst'.
  bool IsNodesDirectlyConnected(const NodeMap& node_map, const string& src,
                                const string& dst, int position = 0);

  // Counts nodes of the given op-type in a graph.
  int CountOpNodes(const GraphDef& graph, const string& op);

  // Get a random tensor with given shape.
  template <DataType DTYPE>
  Tensor GenerateRandomTensor(const TensorShape& shape) const {
    typedef typename EnumToDataType<DTYPE>::Type T;
    Tensor tensor(DTYPE, shape);
    for (auto i = 0; i < tensor.NumElements(); i++)
      tensor.flat<T>()(i) = i + random::New64() % 10;
    return tensor;
  }

  // Creates a random tensor with given shape using `setRandom`.
  template <DataType DTYPE>
  Tensor GenerateTensorWithSetRandom(const TensorShape& shape) const {
    typedef typename EnumToDataType<DTYPE>::Type T;
    Tensor tensor(DTYPE, shape);
    tensor.flat<T>().setRandom();
    return tensor;
  }

  // Get a constant tensor with given shape.
  template <DataType DTYPE>
  Tensor GenerateConstantTensor(
      const TensorShape& shape,
      typename EnumToDataType<DTYPE>::Type value) const {
    typedef typename EnumToDataType<DTYPE>::Type T;
    Tensor tensor(DTYPE, shape);
    for (auto i = 0; i < tensor.NumElements(); i++) tensor.flat<T>()(i) = value;
    return tensor;
  }

  inline machina::Scope CreateScopeWithDevice(absl::string_view device) {
    return machina::Scope::NewRootScope().WithDevice(string(device));
  }

 private:
  SessionOptions options_;
};

}  // end namespace grappler
}  // end namespace machina

#endif  // MACHINA_CORE_GRAPPLER_UTILS_GRAPPLER_TEST_H_
