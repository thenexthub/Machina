/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

static Graph* AdjustContrast(int batches, int width, int height) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor in(DT_FLOAT, TensorShape({batches, width, height, 3}));
  in.flat<float>().setRandom();
  Tensor factor(DT_FLOAT, TensorShape({}));
  factor.flat<float>().setConstant(1.2);

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "AdjustContrastv2")
                  .Input(test::graph::Constant(g, in))
                  .Input(test::graph::Constant(g, factor))
                  .Finalize(g, &ret));
  return g;
}

#define BM_AdjustContrastDev(DEVICE, B, W, H)                    \
  static void BM_AdjustContrast_##DEVICE##_##B##_##W##_##H(      \
      ::testing::benchmark::State& state) {                      \
    test::Benchmark(#DEVICE, AdjustContrast(B, W, H),            \
                    /*old_benchmark_api*/ false)                 \
        .Run(state);                                             \
    state.SetItemsProcessed(state.iterations() * B * W * H * 3); \
  }                                                              \
  BENCHMARK(BM_AdjustContrast_##DEVICE##_##B##_##W##_##H)

// Benchmark results as of cl/106323955
// BM_AdjustContrast_cpu_1_299_299  3416770  22008951  100  11.6M items/s
// BM_AdjustContrast_gpu_32_299_299  37117844  45512374  100  179.8M items/s
// Benchmark results as of cl/109478777
// (note that the definition has changed to perform no min/max or clamping,
// so a comparison to cl/106323955 is inherently unfair)
// The GPU test ran with -c opt --config=cuda --copt=-mavx, CPU ran without
// --config=cuda because for some reason that killed throughput measurement.
// CPU: Intel Haswell with HyperThreading (6 cores) dL1:32KB dL2:256KB dL3:15MB
// GPU: Tesla K40m
// BM_AdjustContrast_cpu_1_299_299     179084     340186  2181  751.9M items/s
// BM_AdjustContrast_gpu_32_299_299     85276     123665  4189  2.9G items/s
BM_AdjustContrastDev(cpu, 1, 299, 299);
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(MACHINA_USE_ROCM) && MACHINA_USE_ROCM)
BM_AdjustContrastDev(gpu, 32, 299, 299);
#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM

}  // namespace machina
