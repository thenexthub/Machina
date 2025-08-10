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

#include <random>

#include "machina/core/common_runtime/kernel_benchmark_testlib.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/platform/test.h"
#include "machina/core/platform/test_benchmark.h"

namespace machina {
namespace {

Tensor VecShape(int64_t v) {
  if (v >= std::numeric_limits<int32>::max()) {
    Tensor shape(DT_INT64, TensorShape({1}));
    shape.vec<int64_t>()(0) = v;
    return shape;
  } else {
    Tensor shape(DT_INT32, TensorShape({1}));
    shape.vec<int32>()(0) = v;
    return shape;
  }
}

Tensor VecLam32(int64_t n, int magnitude) {
  std::mt19937 gen(0x12345);
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  Tensor lams(DT_FLOAT, TensorShape({n}));
  for (int i = 0; i < n; i++) {
    // Generate in range (magnitude, 2 * magnitude)
    lams.vec<float>()(i) = magnitude * (1 + dist(gen));
  }
  return lams;
}

Tensor VecLam64(int64_t n, int magnitude) {
  std::mt19937 gen(0x12345);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  Tensor lams(DT_DOUBLE, TensorShape({n}));
  for (int i = 0; i < n; i++) {
    // Generate in range (magnitude, 2 * magnitude)
    lams.vec<double>()(i) = magnitude * (1 + dist(gen));
  }
  return lams;
}

#define BM_Poisson(DEVICE, BITS, MAGNITUDE)                                    \
  static void BM_##DEVICE##_RandomPoisson_lam_##MAGNITUDE##_##BITS(            \
      ::testing::benchmark::State& state) {                                    \
    const int nsamp = state.range(0);                                          \
    const int nlam = state.range(1);                                           \
                                                                               \
    Graph* g = new Graph(OpRegistry::Global());                                \
    test::graph::RandomPoisson(                                                \
        g, test::graph::Constant(g, VecShape(nsamp)),                          \
        test::graph::Constant(g, VecLam##BITS(nlam, MAGNITUDE)));              \
    test::Benchmark(#DEVICE, g, /*old_benchmark_api*/ false).Run(state);       \
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * nsamp * \
                            nlam);                                             \
  }                                                                            \
  BENCHMARK(BM_##DEVICE##_RandomPoisson_lam_##MAGNITUDE##_##BITS)              \
      ->RangePair(1, 64, 2, 50);

BM_Poisson(cpu, 32, 1);
BM_Poisson(cpu, 32, 8);
BM_Poisson(cpu, 32, 32);

BM_Poisson(cpu, 64, 1);
BM_Poisson(cpu, 64, 8);
BM_Poisson(cpu, 64, 32);

}  // namespace
}  // namespace machina
