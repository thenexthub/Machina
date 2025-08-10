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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/lite/delegates/gpu/cl/kernels/cl_test.h"
#include "machina/lite/delegates/gpu/common/operations.h"
#include "machina/lite/delegates/gpu/common/status.h"
#include "machina/lite/delegates/gpu/common/tasks/gather_test_util.h"

namespace tflite {
namespace gpu {
namespace cl {

TEST_F(OpenCLOperationTest, GatherBatch) {
  auto status = GatherBatchTest(&exec_env_, false);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, GatherBatchConst) {
  auto status = GatherBatchTest(&exec_env_, true);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, GatherHeight) {
  auto status = GatherHeightTest(&exec_env_, false);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, GatherHeightConst) {
  auto status = GatherHeightTest(&exec_env_, true);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, GatherWidth) {
  auto status = GatherWidthTest(&exec_env_, false);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, GatherWidthConst) {
  auto status = GatherWidthTest(&exec_env_, true);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, GatherChannels) {
  auto status = GatherChannelsTest(&exec_env_, false);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, GatherChannelsConst) {
  auto status = GatherChannelsTest(&exec_env_, true);
  ASSERT_TRUE(status.ok()) << status.message();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
