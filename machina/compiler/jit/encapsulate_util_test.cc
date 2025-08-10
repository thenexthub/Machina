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

#include "machina/compiler/jit/encapsulate_util.h"

#include "machina/cc/framework/scope.h"
#include "machina/cc/ops/array_ops.h"
#include "machina/cc/ops/standard_ops.h"
#include "machina/core/framework/node_def_util.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/tensor_shape.pb.h"
#include "machina/core/platform/test.h"

namespace machina {

TEST(PerformStaticShapeInferenceBeforeEncapsulationTest, Basic) {
  // Build the graph:
  // "add" = "const_0" + "const_1"
  // "identity" = "add"
  machina::Scope s = machina::Scope::NewRootScope();
  Output const_0 = ops::Const(s.WithOpName("const_0"), 1, {2});
  Output const_1 = ops::Const(s.WithOpName("const_1"), 2, {2});
  Output add = ops::Add(s.WithOpName("add"), const_0, const_1);
  Output identity = ops::Identity(s.WithOpName("identity"), add);
  Graph g(OpRegistry::Global());
  TF_CHECK_OK(s.ToGraph(&g));

  TF_CHECK_OK(PerformStaticShapeInferenceBeforeEncapsulation(&g));

  // Check that "add" node now has _xla_inferred_shapes attr.
  auto node_index = g.BuildNodeNameIndex();
  Node *add_node = node_index["add"];
  std::vector<PartialTensorShape> output_shapes;
  TF_CHECK_OK(GetNodeAttr(add_node->attrs(), kXlaInferredShapesAttrName,
                          &output_shapes));
  EXPECT_EQ(output_shapes.size(), 1);
  TensorShapeProto shape_proto;
  output_shapes[0].AsProto(&shape_proto);
  EXPECT_EQ(shape_proto.dim_size(), 1);
  EXPECT_EQ(shape_proto.dim(0).size(), 2);
}

}  // namespace machina
