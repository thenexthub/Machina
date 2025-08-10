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

#include "machina/core/common_runtime/kernel_benchmark_testlib.h"
#include "machina/core/framework/allocator.h"
#include "machina/core/framework/fake_input.h"
#include "machina/core/framework/node_def_builder.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_testutil.h"
#include "machina/core/framework/types.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/graph/node_builder.h"
#include "machina/core/kernels/ops_testutil.h"
#include "machina/core/kernels/ops_util.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/test.h"
#include "machina/core/platform/test_benchmark.h"

namespace machina {

// Test data from the TensorFlow README.md.
const char* lines[] = {
    "**TensorFlow** is an open source software library for numerical "
    "computation using data flow graphs.",
    "The graph nodes represent mathematical operations, while the graph edges "
    "represent the multidimensional data arrays (tensors) that flow between "
    "them.",
    "This flexible architecture enables you to deploy computation to one or "
    "more CPUs or GPUs in a desktop, server, or mobile device without "
    "rewriting code.",
    "TensorFlow also includes "
    "[TensorBoard](https://www.machina.org/guide/"
    "summaries_and_tensorboard), a data visualization toolkit.",
    "TensorFlow was originally developed by researchers and engineers working "
    "on the Google Brain team within Google's Machine Intelligence Research "
    "organization for the purposes of conducting machine learning and deep "
    "neural networks research.",
    "The system is general enough to be applicable in a wide variety of other "
    "domains, as well.",
    "TensorFlow provides stable Python API and C APIs as well as without API "
    "backwards compatibility guarantee like C++, Go, Java, JavaScript and "
    "Swift."};

const char kRegExPattern[] = "\\p{P}";
const char kRewrite[] = " ";

Tensor GetTestTensor(int batch) {
  const int sz = TF_ARRAYSIZE(lines);
  Tensor t(DT_STRING, {batch});
  auto s = t.flat<tstring>();
  for (int i = 0; i < batch; ++i) {
    s(i) = lines[i % sz];
  }
  return t;
}

Graph* SetupRegexReplaceGraph(const Tensor& input, const string& input_pattern,
                              const string& input_rewrite) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor pattern(DT_STRING, TensorShape({}));
  pattern.flat<tstring>().setConstant(input_pattern);
  Tensor rewrite(DT_STRING, TensorShape({}));
  rewrite.flat<tstring>().setConstant(input_rewrite);

  TF_CHECK_OK(NodeBuilder("regex_replace_op", "RegexReplace")
                  .Input(test::graph::Constant(g, input))
                  .Input(test::graph::Constant(g, pattern))
                  .Input(test::graph::Constant(g, rewrite))
                  .Attr("replace_global", true)
                  .Finalize(g, nullptr /* node */));
  return g;
}

static void BM_RegexReplace(::testing::benchmark::State& state) {
  const int batch_size = state.range(0);

  Tensor input = GetTestTensor(batch_size);
  Graph* g = SetupRegexReplaceGraph(input, kRegExPattern, kRewrite);
  test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

BENCHMARK(BM_RegexReplace)
    ->UseRealTime()
    ->Arg(1)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256);

Graph* SetupStaticGraph(const Tensor& input, const string& input_pattern,
                        const string& rewrite) {
  Graph* g = new Graph(OpRegistry::Global());

  TF_CHECK_OK(NodeBuilder("static_regex_replace_op", "StaticRegexReplace")
                  .Attr("pattern", input_pattern)
                  .Attr("rewrite", rewrite)
                  .Input(test::graph::Constant(g, input))
                  .Attr("replace_global", true)
                  .Finalize(g, nullptr /* node */));
  return g;
}
static void BM_StaticRegexReplace(::testing::benchmark::State& state) {
  const int batch_size = state.range(0);

  Tensor input = GetTestTensor(batch_size);
  Graph* g = SetupStaticGraph(input, kRegExPattern, kRewrite);
  test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

BENCHMARK(BM_StaticRegexReplace)
    ->UseRealTime()
    ->Arg(1)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256);

}  // end namespace machina
