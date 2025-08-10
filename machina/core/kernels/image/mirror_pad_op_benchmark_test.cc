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

#include "machina/core/common_runtime/kernel_benchmark_testlib.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/graph/node_builder.h"
#include "machina/core/platform/test.h"
#include "machina/core/platform/test_benchmark.h"

namespace machina {

static Graph* MirrorPad(int batches, int height, int width, int depth, int pad,
                        const char* mode) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor in(DT_FLOAT, TensorShape({batches, height, width, depth}));
  in.flat<float>().setRandom();
  Tensor padding(DT_INT32, TensorShape({4, 2}));
  auto boxes_tensor = padding.flat<int>().setZero();
  for (int i = 2; i < 6; i++) boxes_tensor(i) = pad;

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "MirrorPad")
                  .Input(test::graph::Constant(g, in))
                  .Input(test::graph::Constant(g, padding))
                  .Attr("mode", mode)
                  .Finalize(g, &ret));
  return g;
}

#define BM_MirrorPadDev(DEVICE, B, W, H, D, P, MODE)                        \
  static void BM_MirrorPad_##DEVICE##_##B##_##W##_##H##_##D##_##P##_##MODE( \
      ::testing::benchmark::State& state) {                                 \
    test::Benchmark(#DEVICE, MirrorPad(B, W, H, D, P, #MODE),               \
                    /*old_benchmark_api*/ false)                            \
        .Run(state);                                                        \
    state.SetItemsProcessed(state.iterations() * B * (W + 2 * P) *          \
                            (H + 2 * P) * D / 32);                          \
  }                                                                         \
  BENCHMARK(BM_MirrorPad_##DEVICE##_##B##_##W##_##H##_##D##_##P##_##MODE);

BM_MirrorPadDev(cpu, 1, 16, 16, 32, 1, REFLECT);
BM_MirrorPadDev(cpu, 1, 16, 16, 32, 8, REFLECT);
BM_MirrorPadDev(cpu, 1, 512, 512, 16, 1, REFLECT);
BM_MirrorPadDev(cpu, 1, 512, 512, 16, 256, REFLECT);
BM_MirrorPadDev(cpu, 1, 16, 16, 32, 1, SYMMETRIC);
BM_MirrorPadDev(cpu, 1, 16, 16, 32, 8, SYMMETRIC);
BM_MirrorPadDev(cpu, 1, 512, 512, 16, 1, SYMMETRIC);
BM_MirrorPadDev(cpu, 1, 512, 512, 16, 256, SYMMETRIC);

}  // namespace machina
