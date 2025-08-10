/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

static void BM_ExpandDims(::testing::benchmark::State& state) {
  Graph* g = new Graph(OpRegistry::Global());

  Tensor input(DT_INT32, TensorShape({1, 1, 1, 1}));
  input.flat<int32>()(0) = 10;

  Tensor axis(DT_INT32, TensorShape({}));
  axis.flat<int32>()(0) = 2;

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "ExpandDims")
                  .Input(test::graph::Constant(g, input))
                  .Input(test::graph::Constant(g, axis))
                  .Attr("T", DT_INT32)
                  .Attr("Tdim", DT_INT32)
                  .Finalize(g, &node));
  FixupSourceAndSinkEdges(g);

  test::Benchmark("cpu", g, nullptr, nullptr, nullptr,
                  "SINGLE_THREADED_EXECUTOR", /*old_benchmark_api*/ false)
      .Run(state);
}

BENCHMARK(BM_ExpandDims)->UseRealTime();

}  // namespace
}  // namespace machina
