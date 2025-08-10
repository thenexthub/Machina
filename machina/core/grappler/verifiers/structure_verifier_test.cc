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

#include "machina/core/grappler/verifiers/structure_verifier.h"

#include <memory>

#include "absl/strings/match.h"
#include "machina/cc/ops/parsing_ops.h"
#include "machina/cc/ops/standard_ops.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/shape_inference.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/protobuf.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace grappler {
namespace {

class StructureVerifierTest : public ::testing::Test {
 protected:
  StructureVerifierTest() { verifier_ = std::make_unique<StructureVerifier>(); }
  void SetGraph(const string& gdef_ascii) {
    CHECK(protobuf::TextFormat::ParseFromString(gdef_ascii, &graph_));
  }
  GraphDef graph_;
  std::unique_ptr<StructureVerifier> verifier_;
};

absl::Status Scalars(shape_inference::InferenceContext* c) {
  for (int i = 0; i < c->num_outputs(); ++i) {
    c->set_output(i, c->Scalar());
  }
  return absl::OkStatus();
}

REGISTER_OP("TestParams").Output("o: float").SetShapeFn(Scalars);
REGISTER_OP("TestInput")
    .Output("a: float")
    .Output("b: float")
    .SetShapeFn(Scalars);
REGISTER_OP("TestMul")
    .Input("a: float")
    .Input("b: float")
    .Output("o: float")
    .SetShapeFn(Scalars);

TEST_F(StructureVerifierTest, ValidGraphs) {
  // With scope, ops gets registered automatically.
  machina::Scope s = machina::Scope::NewRootScope();
  Output a = ops::Const(s.WithOpName("a"), 0.0f, {10, 10});
  ops::ShapeN b(s.WithOpName("b"), {a, a, a});

  GraphDef graph;
  TF_CHECK_OK(s.ToGraphDef(&graph));
  TF_EXPECT_OK(verifier_->Verify(graph));

  // With graphdef directly, relies on REGISTER_OP to register ops
  SetGraph(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: [ 'W1', 'input:1' ] }");

  TF_EXPECT_OK(verifier_->Verify(graph_));
}

TEST_F(StructureVerifierTest, OpNotRegistered) {
  SetGraph(
      "node { name: 'input' op: 'OpNotRegistered' }"
      "node { name: 't1' op: 'TestMul' input: [ 'input:0', 't2' ] }"
      "node { name: 't2' op: 'TestMul' input: [ 'input:1', 't1' ] }");
  absl::Status status = verifier_->Verify(graph_);
  EXPECT_TRUE(absl::IsNotFound(status));
  EXPECT_TRUE(absl::StrContains(status.message(), "Op type not registered"));
}

TEST_F(StructureVerifierTest, DuplicateNodeNames) {
  SetGraph(
      "node { name: 'A' op: 'TestParams' }"
      "node { name: 'A' op: 'TestInput' }");
  absl::Status status = verifier_->Verify(graph_);
  EXPECT_TRUE(absl::IsAlreadyExists(status));
  EXPECT_TRUE(absl::StrContains(status.message(), "Node already exists:"));
}

TEST_F(StructureVerifierTest, GraphWithInvalidCycle) {
  SetGraph(
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: [ 'input:0', 't2' ] }"
      "node { name: 't2' op: 'TestMul' input: [ 'input:1', 't1' ] }");
  absl::Status status = verifier_->Verify(graph_);
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_TRUE(absl::StrContains(
      status.message(), "The graph couldn't be sorted in topological order"));
}

}  // namespace
}  // namespace grappler
}  // namespace machina
