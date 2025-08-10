/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/lite/delegates/gpu/cl/kernels/cl_test.h"
#include "machina/lite/delegates/gpu/common/operations.h"
#include "machina/lite/delegates/gpu/common/status.h"
#include "machina/lite/delegates/gpu/common/tasks/relu_test_util.h"

namespace tflite {
namespace gpu {
namespace cl {

TEST_F(OpenCLOperationTest, ReLUNoClipNoAlpha) {
  auto status = ReLUNoClipNoAlphaTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, ReLUClip) {
  auto status = ReLUClipTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, ReLUAlpha) {
  auto status = ReLUAlphaTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, ReLUAlphaClip) {
  auto status = ReLUAlphaClipTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, ReLULN1NoClipNoAlpha) {
  auto status = ReLUN1NoClipNoAlphaTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status;
}

TEST_F(OpenCLOperationTest, ReLUN1Clip) {
  auto status = ReLUN1ClipTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status;
}

TEST_F(OpenCLOperationTest, ReLULN1Alpha) {
  auto status = ReLUN1AlphaTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status;
}

TEST_F(OpenCLOperationTest, ReLUN1AlphaClip) {
  auto status = ReLUN1AlphaClipTest(&exec_env_);
  ASSERT_TRUE(status.ok()) << status;
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
