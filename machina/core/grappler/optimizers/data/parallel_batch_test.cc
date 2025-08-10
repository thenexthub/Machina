/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include "machina/core/grappler/optimizers/data/parallel_batch.h"

#include "machina/core/framework/attr_value_util.h"
#include "machina/core/framework/function_testlib.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/tensor_testutil.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/grappler/grappler_item.h"
#include "machina/core/grappler/optimizers/data/graph_test_utils.h"
#include "machina/core/grappler/optimizers/data/graph_utils.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace grappler {
namespace {

TEST(ParallelBatch, BatchDataset) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("batch_size", "Const", {}, {{"value", 5}, {"dtype", DT_INT32}}),
       NDef("drop_remainder", "Const", {},
            {{"value", false}, {"dtype", DT_BOOL}}),
       NDef("batch", "BatchDatasetV2",
            {"range", "batch_size", "drop_remainder"}, {})});

  ParallelBatch optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  EXPECT_TRUE(graph_utils::ContainsGraphNodeWithName("batch", output));
  int index = graph_utils::FindGraphNodeWithName("batch", output);
  EXPECT_TRUE(output.node(index).attr().at("parallel_copy").b());
}

TEST(ParallelBatch, PaddedBatchDataset) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("batch_size", "Const", {}, {{"value", 5}, {"dtype", DT_INT32}}),
       NDef("drop_remainder", "Const", {},
            {{"value", false}, {"dtype", DT_BOOL}}),
       NDef("batch", "PaddedBatchDatasetV2",
            {"range", "batch_size", "drop_remainder"}, {})});

  ParallelBatch optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  EXPECT_TRUE(graph_utils::ContainsGraphNodeWithName("batch", output));
  int index = graph_utils::FindGraphNodeWithName("batch", output);
  EXPECT_TRUE(output.node(index).attr().at("parallel_copy").b());
}

}  // namespace
}  // namespace grappler
}  // namespace machina
