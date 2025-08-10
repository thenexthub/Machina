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

#include "machina/cc/saved_model/experimental/public/saved_model_api.h"

#include <memory>
#include <string>
#include <unordered_set>

#include "machina/c/tf_status.h"
#include "machina/cc/experimental/base/public/runtime.h"
#include "machina/cc/experimental/base/public/runtime_builder.h"
#include "machina/cc/experimental/base/public/status.h"
#include "machina/core/platform/path.h"
#include "machina/core/platform/stringpiece.h"
#include "machina/core/platform/test.h"


namespace {

using machina::experimental::cc::Runtime;
using machina::experimental::cc::RuntimeBuilder;
using machina::experimental::cc::SavedModelAPI;
using machina::experimental::cc::Status;

constexpr char kTestData[] = "cc/saved_model/testdata";

std::string SavedModelPath(absl::string_view saved_model_dir) {
  return machina::io::JoinPath(machina::testing::TensorFlowSrcRoot(),
                                  kTestData, saved_model_dir);
}

// This value parameterized test allows us to test both TFRT
// and non TFRT runtimes.
// https://github.com/google/googletest/blob/dcc92d0ab6c4ce022162a23566d44f673251eee4/googletest/docs/advanced.md#value-parameterized-tests
class CPPSavedModelAPITest : public ::testing::TestWithParam<bool> {};

TEST_P(CPPSavedModelAPITest, LoadsSavedModelWithTags) {
  Status status;
  RuntimeBuilder builder;
  bool use_tfrt = GetParam();
  if (use_tfrt) {
    GTEST_SKIP();  // TODO(chky) : Enable this once TFRT is open sourced.
  }

  builder.SetUseTFRT(use_tfrt);
  std::unique_ptr<Runtime> runtime = builder.Build(&status);
  ASSERT_TRUE(status.ok()) << status.message();

  std::string model_dir = SavedModelPath("VarsAndArithmeticObjectGraph");
  std::unordered_set<std::string> tags = {"serve"};
  std::unique_ptr<SavedModelAPI> model =
      SavedModelAPI::Load(model_dir, *runtime, &status, &tags);

  // TODO(bmzhao): Change this to expect TF_OK when loading is implemented.
  // That unblocks writing other tests that require a TF_SavedModel*,
  // like loading a ConcreteFunction. This test at least checks that the
  // C API builds and can be minimally run.
  EXPECT_EQ(status.code(), TF_UNIMPLEMENTED);
}

TEST_P(CPPSavedModelAPITest, LoadsSavedModel) {
  Status status;
  RuntimeBuilder builder;
  bool use_tfrt = GetParam();
  if (use_tfrt) {
    GTEST_SKIP();  // TODO(chky) : Enable this once TFRT is open sourced.
  }

  builder.SetUseTFRT(use_tfrt);
  std::unique_ptr<Runtime> runtime = builder.Build(&status);
  ASSERT_TRUE(status.ok()) << status.message();

  std::string model_dir = SavedModelPath("VarsAndArithmeticObjectGraph");
  std::unique_ptr<SavedModelAPI> model =
      SavedModelAPI::Load(model_dir, *runtime, &status);

  EXPECT_EQ(status.code(), TF_OK) << status.message();
}

INSTANTIATE_TEST_SUITE_P(RuntimeAgnosticCPPSavedModelTests,
                         CPPSavedModelAPITest, ::testing::Bool());

}  // namespace

