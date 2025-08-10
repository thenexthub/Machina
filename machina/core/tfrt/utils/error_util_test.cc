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
#include "machina/core/tfrt/utils/error_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "machina/xla/tsl/concurrency/async_value_ref.h"
#include "machina/core/platform/status.h"
#include "tfrt/support/error_util.h"  // from @tf_runtime

namespace tfrt {
namespace {

TEST(ErrorUtilTest, AllSupportedErrorConversion){
#define ERROR_TYPE(TFRT_ERROR, TF_ERROR)                                 \
  {                                                                      \
    machina::Status status(absl::StatusCode::TF_ERROR, "error_test"); \
    EXPECT_EQ(ConvertTfErrorCodeToTfrtErrorCode(status),                 \
              tfrt::ErrorCode::TFRT_ERROR);                              \
  }
#include "machina/core/tfrt/utils/error_type.def"  // NOLINT
}

TEST(ErrorUtilTest, UnsupportedErrorConversion) {
  absl::Status status(absl::StatusCode::kUnauthenticated, "error_test");
  EXPECT_EQ(ConvertTfErrorCodeToTfrtErrorCode(status),
            tfrt::ErrorCode::kUnknown);
}

TEST(ErrorUtilTest, ToTfStatusError) {
  auto error_av =
      tsl::MakeErrorAsyncValueRef(absl::UnauthenticatedError("test_error"));
  auto status = ToTfStatus(error_av.get());
  EXPECT_EQ(status.code(), absl::StatusCode::kUnauthenticated);
  EXPECT_EQ(status.message(), "test_error");
}

}  // namespace
}  // namespace tfrt
