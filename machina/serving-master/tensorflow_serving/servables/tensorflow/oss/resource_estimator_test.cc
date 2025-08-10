/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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
#include "machina_serving/servables/machina/oss/resource_estimator.h"

#include <cstddef>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/path.h"
#include "machina_serving/servables/machina/bundle_factory_test_util.h"
#include "machina_serving/test_util/test_util.h"
#include "machina_serving/util/test_util/mock_file_probing_env.h"

namespace machina {
namespace serving {
namespace {

using test_util::EqualsProto;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

class ResourceEstimatorTest : public ::testing::Test {
 protected:
  void SetUp() {
    export_dir_ = "/foo/bar";
    const string child = "child";
    const string child_path = io::JoinPath(export_dir_, child);
    file_size_ = 100;

    // Set up the expectation that the directory contains exactly one child with
    // the given file size.
    EXPECT_CALL(env_, FileExists(export_dir_))
        .WillRepeatedly(Return(absl::Status()));
    EXPECT_CALL(env_, GetChildren(export_dir_, _))
        .WillRepeatedly(DoAll(SetArgPointee<1>(std::vector<string>({child})),
                              Return(absl::Status())));
    EXPECT_CALL(env_, IsDirectory(child_path))
        .WillRepeatedly(Return(errors::FailedPrecondition("")));
    EXPECT_CALL(env_, GetFileSize(child_path, _))
        .WillRepeatedly(
            DoAll(SetArgPointee<1>(file_size_), Return(absl::Status())));
  }

  string export_dir_;
  double file_size_;
  test_util::MockFileProbingEnv env_;
};

TEST_F(ResourceEstimatorTest, EstimateResourceFromPathWithFileProbingEnv) {
  ResourceAllocation actual;
  TF_ASSERT_OK(EstimateMainRamBytesFromPath(
      export_dir_, /*use_validation_result=*/false, &env_, &actual));
  ResourceAllocation expected =
      test_util::GetExpectedResourceEstimate(file_size_);
  EXPECT_THAT(actual, EqualsProto(expected));
}

TEST_F(ResourceEstimatorTest, EstimateResourceFromValidationResult) {
  // Currently, using validation result is not supported yet.
  // Uses disk state to estimate the resource usage.
  ResourceAllocation actual;
  TF_ASSERT_OK(EstimateMainRamBytesFromPath(
      export_dir_, /*use_validation_result=*/true, &env_, &actual));
  ResourceAllocation expected =
      test_util::GetExpectedResourceEstimate(file_size_);
  EXPECT_THAT(actual, EqualsProto(expected));
}

}  // namespace
}  // namespace serving
}  // namespace machina
