/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#include "machina/lite/delegates/gpu/gl/android_sync.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/lite/delegates/gpu/gl/egl_environment.h"

namespace tflite::gpu::gl {

// Make sure GPU fences can be waited on by the GPU
TEST(AsyncBufferTest, FenceTest) {
  // Check falseness first
  EXPECT_EQ(CreateFdGpu(), -1);
  EXPECT_FALSE(WaitFdGpu(1));  // False because EGL isn't set up
  std::unique_ptr<EglEnvironment> env;
  EXPECT_OK(EglEnvironment::NewEglEnvironment(&env));
  int gpu_fd = CreateFdGpu();
  EXPECT_GE(gpu_fd, 0);
  EXPECT_TRUE(WaitFdGpu(gpu_fd));
}

}  // namespace tflite::gpu::gl
