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

#ifndef MACHINA_LITE_DELEGATES_GPU_CL_CL_TEST_H_
#define MACHINA_LITE_DELEGATES_GPU_CL_CL_TEST_H_

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/lite/delegates/gpu/cl/environment.h"
#include "machina/lite/delegates/gpu/cl/opencl_wrapper.h"

namespace tflite {
namespace gpu {
namespace cl {

#ifndef ASSERT_OK
#define ASSERT_OK(x) ASSERT_TRUE(x.ok());
#endif

class OpenCLTest : public ::testing::Test {
 public:
  void SetUp() override {
    ASSERT_OK(LoadOpenCL());
    ASSERT_OK(CreateEnvironment(&env_));
  }

 protected:
  Environment env_;
};

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_CL_CL_TEST_H_
