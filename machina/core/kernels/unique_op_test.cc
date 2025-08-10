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
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor.pb.h"
#include "machina/core/framework/tensor_shape.pb.h"
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

const int kMaxStrLen = 40;

TensorProto GetRandomInt32TensorProto(int dim, int max_int) {
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_INT32);
  tensor_proto.mutable_tensor_shape()->add_dim()->set_size(dim);
  tensor_proto.mutable_tensor_shape()->set_unknown_rank(false);
  for (int i = 0; i < dim; ++i) {
    const int int_val = std::rand() % max_int;
    tensor_proto.add_int_val(int_val);
  }
  return tensor_proto;
}

TensorProto GetRandomInt32TensorProtoWithRepeat(int dim, int repeat,
                                                int max_int) {
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_INT32);
  tensor_proto.mutable_tensor_shape()->add_dim()->set_size(dim);
  tensor_proto.mutable_tensor_shape()->set_unknown_rank(false);
  for (int i = 0; i < dim; ++i) {
    const int int_val = std::rand() % max_int;
    for (int j = 0; j < repeat; ++j) {
      tensor_proto.add_int_val(int_val);
    }
  }
  return tensor_proto;
}

void BM_Unique_INT32(::testing::benchmark::State& state) {
  const int dim = state.range(0);
  const int max_int = state.range(1);

  Graph* g = new Graph(OpRegistry::Global());

  Tensor input(DT_INT32, TensorShape({dim}));
  CHECK(input.FromProto(GetRandomInt32TensorProto(dim, max_int)));

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Unique")
                  .Input(test::graph::Constant(g, input))
                  .Attr("T", DT_INT32)
                  .Finalize(g, &node));
  FixupSourceAndSinkEdges(g);

  test::Benchmark("cpu", g, nullptr, nullptr, nullptr,
                  "SINGLE_THREADED_EXECUTOR", /*old_benchmark_api*/ false)
      .Run(state);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * dim *
                          sizeof(int32));
}

void BM_Unique_INT32_Repeat(::testing::benchmark::State& state) {
  const int dim = state.range(0);
  const int max_int = state.range(1);

  Graph* g = new Graph(OpRegistry::Global());

  Tensor input(DT_INT32, TensorShape({dim * 200}));
  CHECK(
      input.FromProto(GetRandomInt32TensorProtoWithRepeat(dim, 200, max_int)));

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Unique")
                  .Input(test::graph::Constant(g, input))
                  .Attr("T", DT_INT32)
                  .Finalize(g, &node));
  FixupSourceAndSinkEdges(g);

  test::Benchmark("cpu", g, nullptr, nullptr, nullptr,
                  "SINGLE_THREADED_EXECUTOR", /*old_benchmark_api*/ false)
      .Run(state);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * dim * 200 *
                          sizeof(int32));
}

TensorProto GetRandomStringsTensorProto(int dim, int max_str_len) {
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_STRING);
  tensor_proto.mutable_tensor_shape()->add_dim()->set_size(dim);
  tensor_proto.mutable_tensor_shape()->set_unknown_rank(false);
  for (int i = 0; i < dim; ++i) {
    const int len = std::rand() % max_str_len + 1;
    string rand_str;
    rand_str.resize(len);
    for (int j = 0; j < len; ++j) {
      rand_str[j] = static_cast<char>(j % 256);
    }
    tensor_proto.add_string_val(rand_str);
  }
  return tensor_proto;
}

void BM_Unique_STRING(::testing::benchmark::State& state) {
  const int dim = state.range(0);

  Graph* g = new Graph(OpRegistry::Global());

  Tensor input(DT_STRING, TensorShape({dim}));
  CHECK(input.FromProto(GetRandomStringsTensorProto(dim, kMaxStrLen)));

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Unique")
                  .Input(test::graph::Constant(g, input))
                  .Attr("T", DT_STRING)
                  .Finalize(g, &node));
  FixupSourceAndSinkEdges(g);
  test::Benchmark("cpu", g, nullptr, nullptr, nullptr,
                  "SINGLE_THREADED_EXECUTOR", /*old_benchmark_api*/ false)
      .Run(state);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * dim *
                          sizeof(tstring));
}

BENCHMARK(BM_Unique_INT32)
    ->UseRealTime()
    ->ArgPair(32, 1024 * 1024)
    ->ArgPair(256, 1024 * 1024)
    ->ArgPair(1024, 1024 * 1024)
    ->ArgPair(4 * 1024, 1024 * 1024)
    ->ArgPair(16 * 1024, 1024 * 1024)
    ->ArgPair(64 * 1024, 1024 * 1024)
    ->ArgPair(1024 * 1024, 1024 * 1024)
    ->ArgPair(4 * 1024 * 1024, 1024 * 1024)
    ->ArgPair(32, 64 * 1024 * 1024)
    ->ArgPair(256, 64 * 1024 * 1024)
    ->ArgPair(1024, 64 * 1024 * 1024)
    ->ArgPair(4 * 1024, 64 * 1024 * 1024)
    ->ArgPair(16 * 1024, 64 * 1024 * 1024)
    ->ArgPair(64 * 1024, 64 * 1024 * 1024)
    ->ArgPair(1024 * 1024, 64 * 1024 * 1024)
    ->ArgPair(4 * 1024 * 1024, 64 * 1024 * 1024);

BENCHMARK(BM_Unique_INT32_Repeat)
    ->UseRealTime()
    ->ArgPair(32, 1024 * 1024)
    ->ArgPair(256, 1024 * 1024)
    ->ArgPair(1024, 1024 * 1024)
    ->ArgPair(4 * 1024, 1024 * 1024)
    ->ArgPair(16 * 1024, 1024 * 1024)
    ->ArgPair(64 * 1024, 1024 * 1024)
    ->ArgPair(1024 * 1024, 1024 * 1024)
    ->ArgPair(4 * 1024 * 1024, 1024 * 1024)
    ->ArgPair(32, 32 * 1024 * 1024)
    ->ArgPair(256, 32 * 1024 * 1024)
    ->ArgPair(1024, 32 * 1024 * 1024)
    ->ArgPair(4 * 1024, 32 * 1024 * 1024)
    ->ArgPair(16 * 1024, 32 * 1024 * 1024)
    ->ArgPair(64 * 1024, 32 * 1024 * 1024)
    ->ArgPair(1024 * 1024, 32 * 1024 * 1024)
    ->ArgPair(32, 64 * 1024 * 1024)
    ->ArgPair(256, 64 * 1024 * 1024)
    ->ArgPair(1024, 64 * 1024 * 1024)
    ->ArgPair(4 * 1024, 64 * 1024 * 1024)
    ->ArgPair(16 * 1024, 64 * 1024 * 1024)
    ->ArgPair(64 * 1024, 64 * 1024 * 1024)
    ->ArgPair(1024 * 1024, 64 * 1024 * 1024);

BENCHMARK(BM_Unique_STRING)
    ->UseRealTime()
    ->Arg(32)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(4 * 1024)
    ->Arg(16 * 1024)
    ->Arg(64 * 1024)
    ->Arg(256 * 1024);

}  // namespace
}  // namespace machina
