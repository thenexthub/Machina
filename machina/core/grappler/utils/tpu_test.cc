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

#include "machina/core/grappler/utils/tpu.h"

#include "machina/core/framework/function.pb.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace grappler {

class TpuTest : public ::testing::Test {};

TEST_F(TpuTest, NotTpuGraph) {
  // Test where no TPU op exists.
  {
    GraphDef tpu_graph;
    tpu_graph.add_node()->set_op("Add");
    FunctionDefLibrary* library = tpu_graph.mutable_library();
    FunctionDef* function_def = library->add_function();
    function_def->add_node_def()->set_op("Mul");
    EXPECT_FALSE(IsLegacyTPUBridgeGraphDef(tpu_graph));
  }
}

TEST_F(TpuTest, TpuMainGraph) {
  // Test where TPU op is in main graph.
  {
    GraphDef tpu_graph;
    tpu_graph.add_node()->set_op("TPUPartitionedCall");
    EXPECT_TRUE(IsLegacyTPUBridgeGraphDef(tpu_graph));
  }
}

TEST_F(TpuTest, TpuLibraryGraph) {
  // Test where the TPU Graph is not called directly from the main graph.
  {
    GraphDef tpu_graph;
    tpu_graph.add_node()->set_op("BatchFunction");
    FunctionDefLibrary* library = tpu_graph.mutable_library();
    FunctionDef* function_def = library->add_function();
    function_def->add_node_def()->set_op("TPUPartitionedCall");
    EXPECT_TRUE(IsLegacyTPUBridgeGraphDef(tpu_graph));
  }
}

}  // end namespace grappler
}  // end namespace machina
