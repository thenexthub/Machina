/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#include "machina/c/experimental/ops/gen/cpp/cpp_generator.h"

#include <algorithm>
#include <vector>

#include "machina/c/experimental/ops/gen/common/path_config.h"
#include "machina/c/experimental/ops/gen/cpp/renderers/cpp_config.h"
#include "machina/xla/tsl/platform/status.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/path.h"
#include "machina/core/platform/test.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace generator {
namespace {

TEST(CppGeneratorTest, typical_usage) {
  string category = "testing";
  string name_space = "machina::ops";
  string output_dir = "machina/c/experimental/ops/gen/cpp/golden";
  string source_dir = "machina";
  string api_dirs = "";
  std::vector<string> ops = {
      "Neg",        // Simple unary Op
      "MatMul",     // 2 inputs & attrs with default values
      "IdentityN",  // Variadic input+output
      "SparseSoftmaxCrossEntropyWithLogits",  // 2 outputs
      "AccumulatorApplyGradient",             // 0 outputs
      "VarHandleOp",                          // type, shape, list(string) attrs
      "RestoreV2",  // Variadic output-only, list(type) attr
  };

  cpp::CppConfig cpp_config(category, name_space);
  PathConfig controller_config(output_dir, source_dir, api_dirs, ops);
  CppGenerator generator(cpp_config, controller_config);

  Env *env = Env::Default();
  string golden_dir = io::JoinPath(testing::TensorFlowSrcRoot(),
                                   controller_config.tf_output_dir);

  string generated_header = generator.HeaderFileContents().Render();
  string generated_source = generator.SourceFileContents().Render();
  string expected_header;
  string header_file_name = io::JoinPath(golden_dir, "testing_ops.h.golden");
  TF_CHECK_OK(ReadFileToString(env, header_file_name, &expected_header));

  string expected_source;
  string source_file_name = io::JoinPath(golden_dir, "testing_ops.cc.golden");
  TF_CHECK_OK(ReadFileToString(env, source_file_name, &expected_source));

  // Remove carriage returns (for Windows)
  expected_header.erase(
      std::remove(expected_header.begin(), expected_header.end(), '\r'),
      expected_header.end());
  expected_source.erase(
      std::remove(expected_source.begin(), expected_source.end(), '\r'),
      expected_source.end());

  EXPECT_EQ(expected_header, generated_header);
  EXPECT_EQ(expected_source, generated_source);
}

}  // namespace
}  // namespace generator
}  // namespace machina
