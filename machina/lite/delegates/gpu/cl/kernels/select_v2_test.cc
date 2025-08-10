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

#include <gtest/gtest.h>
#include "machina/lite/delegates/gpu/cl/kernels/cl_test.h"
#include "machina/lite/delegates/gpu/common/tasks/select_v2_test_util.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

TEST_F(OpenCLOperationTest, SelectV2) {
  auto status = SelectV2Test(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, SelectV2Batch) {
  auto status = SelectV2BatchTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, SelectV2Channels) {
  auto status = SelectV2ChannelsTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, SelectV2ChannelsBatch) {
  auto status = SelectV2ChannelsBatchTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, SelectV2BroadcastTrue) {
  auto status = SelectV2BroadcastTrueTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, SelectV2BroadcastFalse) {
  auto status = SelectV2BroadcastFalseTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, SelectV2BroadcastBoth) {
  auto status = SelectV2BroadcastBothTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, SelectV2ChannelsBroadcastFalse) {
  auto status = SelectV2ChannelsBroadcastFalseTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
