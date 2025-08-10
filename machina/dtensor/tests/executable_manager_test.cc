/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/xla/tsl/platform/status_matchers.h"
#include "machina/xla/tsl/protobuf/error_codes.pb.h"
#include "machina/core/framework/attr_value.pb.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/platform/refcount.h"
#include "machina/dtensor/cc/dtensor_device_util.h"
#include "machina/dtensor/cc/dtensor_operation.h"
#include "machina/dtensor/cc/tensor_layout.h"

namespace machina {
namespace dtensor {
namespace {

using ::testing::HasSubstr;
using ::tsl::error::UNAVAILABLE;
using ::tsl::testing::StatusIs;

class ExecutableManagerTest : public ::testing::Test {
 protected:
  DTensorOperation CreateTestDTensorOperation() {
    return DTensorOperation{"test_fn", nullptr, empty_mesh_, {}};
  }

  Mesh empty_mesh_ = Mesh::Empty();

  core::RefCountPtr<ExecutableManager<ExecutionFunctions>> function_manager_{
      new ExecutableManager<ExecutionFunctions>()};
};

TEST_F(ExecutableManagerTest, ShouldFoldInputUnavailable) {
  auto result =
      function_manager_->ShouldFoldInput(CreateTestDTensorOperation(), {}, 0);
  EXPECT_THAT(result,
              absl_testing::StatusIs(
                  UNAVAILABLE, HasSubstr("ExecutionFunctions manager can not "
                                         "check if the input is foldable")));
}

TEST_F(ExecutableManagerTest, GetCachedExecutableUnavailable) {
  DTensorOperation doperation = CreateTestDTensorOperation();
  NameAttrList func_attr;
  func_attr.set_name(doperation.name);
  auto result = function_manager_->GetCachedExecutable(
      doperation, func_attr,
      {nullptr},  // Dummy input to trigger ShouldFoldInput check.
      {});
  EXPECT_THAT(result, absl_testing::StatusIs(UNAVAILABLE));
}

}  // namespace
}  // namespace dtensor
}  // namespace machina
