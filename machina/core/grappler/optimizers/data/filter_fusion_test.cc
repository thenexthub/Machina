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

#include "machina/core/grappler/optimizers/data/filter_fusion.h"

#include "machina/core/framework/attr_value_util.h"
#include "machina/core/framework/function_testlib.h"
#include "machina/core/framework/tensor_testutil.h"
#include "machina/core/grappler/grappler_item.h"
#include "machina/core/grappler/optimizers/data/graph_test_utils.h"
#include "machina/core/grappler/optimizers/data/graph_utils.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace grappler {
namespace {

using graph_tests_utils::MakeFilterNode;

TEST(FilterFusionTest, FuseTwoFilterIntoOne) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       MakeFilterNode("filter1", "range"),
       MakeFilterNode("filter2", "filter1")},
      // FunctionLib
      {
          test::function::IsZero(),
      });

  FilterFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("FilterDataset", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("filter1", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("filter2", output));
}

TEST(FilterFusionTest, FuseThreeNodesIntoOne) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("filename", "Const", {}, {{"value", ""}, {"dtype", DT_STRING}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       MakeFilterNode("filter1", "range"), MakeFilterNode("filter2", "filter1"),
       MakeFilterNode("filter3", "filter2"),
       NDef("cache", "CacheDataset", {"filter3", "filename"}, {})},
      // FunctionLib
      {
          test::function::IsZero(),
      });

  FilterFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("FilterDataset", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("filter1", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("filter2", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("filter3", output));
}

}  // namespace
}  // namespace grappler
}  // namespace machina
