/* Copyright 2018 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "machina/cc/framework/ops.h"
#include "machina/cc/framework/scope.h"
#include "machina/cc/ops/array_ops.h"
#include "machina/cc/ops/no_op.h"
#include "machina/cc/ops/parsing_ops.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/kernels/fuzzing/fuzz_session.h"

namespace machina {
namespace fuzzing {

class FuzzDecodeCompressed : public FuzzStringInputOp {
  void BuildGraph(const Scope& scope) override {
    auto input =
        machina::ops::Placeholder(scope.WithOpName("input"), DT_STRING);
    auto d1 = machina::ops::DecodeCompressed(
        scope.WithOpName("d1"), input,
        machina::ops::DecodeCompressed::CompressionType(""));
    auto d2 = machina::ops::DecodeCompressed(
        scope.WithOpName("d2"), input,
        machina::ops::DecodeCompressed::CompressionType("ZLIB"));
    auto d3 = machina::ops::DecodeCompressed(
        scope.WithOpName("d3"), input,
        machina::ops::DecodeCompressed::CompressionType("GZIP"));
    Scope grouper =
        scope.WithControlDependencies(std::vector<machina::Operation>{
            d1.output.op(), d2.output.op(), d3.output.op()});
    (void)machina::ops::NoOp(grouper.WithOpName("output"));
  }
};

STANDARD_TF_FUZZ_FUNCTION(FuzzDecodeCompressed);

}  // namespace fuzzing
}  // namespace machina
