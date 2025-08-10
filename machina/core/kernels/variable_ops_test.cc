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

#include "machina/core/framework/op.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/graph/graph.h"
#include "machina/core/graph/node_builder.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/test.h"
#include "machina/core/platform/test_benchmark.h"
#include "machina/core/public/session.h"

namespace machina {
namespace {

// Benchmark to simulate the overhead in training and serving workloads from too
// many threads grabbing the ResourceMgr lock at the same time because of the
// variable and queue ops.
void ManyManyVariablesHelper(int threads, int variables,
                             ::testing::benchmark::State& state) {
  Graph g(OpRegistry::Global());
  std::vector<string> targets;
  for (int i = 0; i < variables; ++i) {
    Node* v;
    TF_CHECK_OK(
        NodeBuilder(
            g.NewName("VeryVeryLongRealistSoundingVariableName/weights"),
            "VariableV2")
            .Attr("shape", TensorShape())
            .Attr("dtype", DT_FLOAT)
            .Finalize(&g, &v));
    targets.push_back(v->name());
  }
  GraphDef gd;
  g.ToGraphDef(&gd);
  SessionOptions opts;
  opts.config.set_inter_op_parallelism_threads(threads);
  Session* sess = NewSession(opts);
  TF_CHECK_OK(sess->Create(gd));
  TF_CHECK_OK(sess->Run({}, {}, targets, nullptr));
  for (auto s : state) {
    TF_CHECK_OK(sess->Run({}, {}, targets, nullptr));
  }
  delete sess;
}

void BM_ManyManyVariablesManyThreads(::testing::benchmark::State& state) {
  const int threads = state.range(0);

  ManyManyVariablesHelper(threads, 1000, state);
}

BENCHMARK(BM_ManyManyVariablesManyThreads)->Arg(50);

}  // namespace
}  // namespace machina
