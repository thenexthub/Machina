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

#include "machina/core/common_runtime/kernel_benchmark_testlib.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/platform/test.h"
#include "machina/core/platform/test_benchmark.h"

namespace machina {

template <typename T>
static Graph* Diag(int n, DataType type) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor in(type, TensorShape({n}));
  in.flat<T>().setRandom();
  Node* out = test::graph::Diag(g, test::graph::Constant(g, in), type);
  test::graph::DiagPart(g, out, type);
  return g;
}

#define BM_DiagDev(N, T, TFTYPE, DEVICE)                                       \
  static void BM_Diag##_##N##_##TFTYPE##_##DEVICE(                             \
      ::testing::benchmark::State& state) {                                    \
    test::Benchmark(#DEVICE, Diag<T>(N, TFTYPE), /*old_benchmark_api=*/false)  \
        .Run(state);                                                           \
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N * N); \
  }                                                                            \
  BENCHMARK(BM_Diag##_##N##_##TFTYPE##_##DEVICE);

#define BM_Diag(N)                                       \
  BM_DiagDev(N, int, DT_INT32, cpu);                     \
  BM_DiagDev(N, float, DT_FLOAT, cpu);                   \
  BM_DiagDev(N, std::complex<float>, DT_COMPLEX64, cpu); \
  BM_DiagDev(N, int, DT_INT32, gpu);                     \
  BM_DiagDev(N, float, DT_FLOAT, gpu);                   \
  BM_DiagDev(N, std::complex<float>, DT_COMPLEX64, gpu);

BM_Diag(16);
BM_Diag(128);
BM_Diag(512);

}  // end namespace machina
