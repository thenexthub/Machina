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

#import <XCTest/XCTest.h>

#include <vector>

#include "machina/lite/delegates/gpu/common/shape.h"
#include "machina/lite/delegates/gpu/common/status.h"
#include "machina/lite/delegates/gpu/common/tasks/resampler_test_util.h"
#include "machina/lite/delegates/gpu/metal/kernels/test_util.h"

@interface ResamplerTest : XCTestCase
@end

@implementation ResamplerTest {
  tflite::gpu::metal::MetalExecutionEnvironment exec_env_;
}

- (void)testResamplerIdentity {
  auto status = ResamplerIdentityTest(tflite::gpu::BHWC(1, 2, 2, 1), &exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());

  status = ResamplerIdentityTest(tflite::gpu::BHWC(1, 3, 5, 3), &exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());

  status = ResamplerIdentityTest(tflite::gpu::BHWC(1, 6, 1, 7), &exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

@end
