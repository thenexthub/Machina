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

#include <functional>
#include <memory>

#include "machina/core/common_runtime/kernel_benchmark_testlib.h"
#include "machina/core/framework/allocator.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/types.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/graph/algorithm.h"
#include "machina/core/graph/node_builder.h"
#include "machina/core/graph/testlib.h"
#include "machina/core/kernels/ops_testutil.h"
#include "machina/core/kernels/ops_util.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/test.h"
#include "machina/core/platform/test_benchmark.h"

namespace machina {
namespace {

// For the benchmark, we set up two 2-dimensional tensors, each kDim1 x 'dim'
// in size, and concat them together along "concat_dimension"
template <typename T>
static void SliceHelper(::testing::benchmark::State& state) {
  const int size = state.range(0);
  Graph* g = new Graph(OpRegistry::Global());
  DataType dt = DataTypeToEnum<T>::v();
  int kDim = 100;
  int kMaxSize = 15000;
  CHECK_LT(size, kMaxSize);

  Tensor begin(DT_INT32, TensorShape({2}));
  begin.flat<int32>()(0) = 10;
  begin.flat<int32>()(1) = 10;

  Tensor sizes(DT_INT32, TensorShape({2}));
  sizes.flat<int32>()(0) = kDim;
  sizes.flat<int32>()(1) = size;

  Tensor input(dt, TensorShape({2 * kDim, kMaxSize}));
  input.flat<T>().setRandom();

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Slice")
                  .Input(test::graph::Constant(g, input))
                  .Input(test::graph::Constant(g, begin))
                  .Input(test::graph::Constant(g, sizes))
                  .Attr("T", dt)
                  .Finalize(g, &node));
  FixupSourceAndSinkEdges(g);

  test::Benchmark("cpu", g, nullptr, nullptr, nullptr,
                  "SINGLE_THREADED_EXECUTOR", /*old_benchmark_api*/ false)
      .Run(state);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * kDim *
                          size * sizeof(T));
}

void BM_SliceFloat(::testing::benchmark::State& state) {
  SliceHelper<float>(state);
}

BENCHMARK(BM_SliceFloat)->UseRealTime()->Arg(100)->Arg(1000)->Arg(10000);

void BM_SliceBFloat16(::testing::benchmark::State& state) {
  SliceHelper<bfloat16>(state);
}

BENCHMARK(BM_SliceBFloat16)->UseRealTime()->Arg(100)->Arg(1000)->Arg(10000);

}  // namespace
}  // namespace machina
