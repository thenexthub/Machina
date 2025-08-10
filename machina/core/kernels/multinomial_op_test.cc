/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#include <functional>
#include <memory>
#include <vector>

#include "machina/core/common_runtime/kernel_benchmark_testlib.h"
#include "machina/core/graph/node_builder.h"
#include "machina/core/kernels/ops_testutil.h"
#include "machina/core/platform/test_benchmark.h"

namespace machina {

static Graph* Multinomial(int batch_size, int num_classes, int num_samples) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor logits_t(DT_FLOAT, TensorShape({batch_size, num_classes}));
  Tensor num_samples_t(DT_INT32, TensorShape());
  logits_t.flat<float>().setRandom();
  num_samples_t.scalar<int32>().setConstant(num_samples);

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("multinomial"), "Multinomial")
                  .Input(test::graph::Constant(g, logits_t))
                  .Input(test::graph::Constant(g, num_samples_t))
                  .Attr("T", DT_FLOAT)
                  .Finalize(g, &ret));
  return g;
}

#define BM_MultinomialDev(DEVICE, B, C, S)                    \
  static void BM_Multinomial_##DEVICE##_##B##_##C##_##S(      \
      ::testing::benchmark::State& state) {                   \
    test::Benchmark(#DEVICE, Multinomial(B, C, S),            \
                    /*old_benchmark_api*/ false)              \
        .Run(state);                                          \
    state.SetItemsProcessed(static_cast<int64_t>(B) * C * S * \
                            state.iterations());              \
  }                                                           \
  BENCHMARK(BM_Multinomial_##DEVICE##_##B##_##C##_##S);

#define BM_MultinomialBCS(B, C, S) \
  BM_MultinomialDev(cpu, B, C, S); \
  BM_MultinomialDev(gpu, B, C, S);

// NOTE(zongheng): some more expensive cases are commented out; left here since
// they could still be useful in covering realistic cases.

BM_MultinomialBCS(1, 10000, 4);
BM_MultinomialBCS(1, 10000, 128);
BM_MultinomialBCS(1, 10000, 10000);
BM_MultinomialBCS(1, 100000, 4);
// BM_MultinomialBCS(1, 100000, 128);

BM_MultinomialBCS(32, 10000, 4);
BM_MultinomialBCS(32, 10000, 128);
BM_MultinomialBCS(32, 100000, 4);
// BM_MultinomialBCS(32, 100000, 128);

BM_MultinomialBCS(128, 100000, 1);
// BM_MultinomialBCS(128, 100000, 128);

}  // namespace machina
