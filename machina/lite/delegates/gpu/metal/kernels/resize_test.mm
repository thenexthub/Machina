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

#import <XCTest/XCTest.h>

#include "machina/lite/delegates/gpu/common/status.h"
#include "machina/lite/delegates/gpu/common/tasks/resize_test_util.h"
#include "machina/lite/delegates/gpu/metal/kernels/test_util.h"

@interface ResizeTest : XCTestCase
@end

@implementation ResizeTest {
  tflite::gpu::metal::MetalExecutionEnvironment exec_env_;
}

- (void)testResizeBilinearAligned {
  auto status = ResizeBilinearAlignedTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testResizeBilinearNonAligned {
  auto status = ResizeBilinearNonAlignedTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testResizeBilinearWithoutHalfPixel {
  auto status = ResizeBilinearWithoutHalfPixelTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testResizeBilinearWithHalfPixel {
  auto status = ResizeBilinearWithHalfPixelTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testResizeNearest {
  auto status = ResizeNearestTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testResizeNearestAlignCorners {
  auto status = ResizeNearestAlignCornersTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testResizeNearestHalfPixelCenters {
  auto status = ResizeNearestHalfPixelCentersTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

@end
