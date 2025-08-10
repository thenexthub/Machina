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

#include "machina_serving/util/retrier.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "machina/xla/tsl/lib/core/status_test_util.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/errors.h"

namespace machina {
namespace serving {
namespace {

using ::testing::HasSubstr;

TEST(RetrierTest, RetryFinallySucceeds) {
  int call_count = 0;
  auto retried_fn = [&]() {
    ++call_count;
    if (call_count == 7) {
      return absl::OkStatus();
    }
    return errors::Unknown("Error");
  };

  TF_EXPECT_OK(Retry("RetryFinallySucceeds", 10 /* max_num_retries */,
                     1 /* retry_interval_micros */, retried_fn));
  EXPECT_EQ(7, call_count);
}

TEST(RetrierTest, RetryFinallyFails) {
  int call_count = 0;
  auto retried_fn = [&]() {
    ++call_count;
    return errors::Unknown("Error");
  };

  const auto status = Retry("RetryFinallyFails", 10 /* max_num_retries */,
                            0 /* retry_interval_micros */, retried_fn);
  EXPECT_THAT(status.message(), HasSubstr("Error"));
  EXPECT_EQ(11, call_count);
}

TEST(RetrierTest, RetryCancelled) {
  int call_count = 0;
  auto retried_fn = [&]() {
    ++call_count;
    return errors::Unknown("Error");
  };
  const auto status = Retry(
      "RetryCancelled", 10 /* max_num_retries */, 0 /* retry_interval_micros */,
      retried_fn, [](absl::Status status) { return false; } /* should retry */);
  EXPECT_THAT(status.message(), HasSubstr("Error"));
  EXPECT_EQ(1, call_count);
}

TEST(RetrierTest, RetryCancelledOnUnimplementedError) {
  int call_count = 0;
  auto retried_fn = [&]() {
    ++call_count;
    if (call_count == 5) {
      return errors::Unimplemented("Unimplemented");
    }
    return errors::DeadlineExceeded("DeadlineExceeded");
  };

  const auto status =
      Retry("RetryCancelledOnUnimplementedError", 10 /* max_num_retries */,
            0 /* retry_interval_micros */, retried_fn, [](absl::Status status) {
              return status.code() != absl::StatusCode::kUnimplemented;
            });
  EXPECT_EQ(5, call_count);
  EXPECT_THAT(status.message(), HasSubstr("Unimplemented"));
}

}  // namespace
}  // namespace serving
}  // namespace machina
