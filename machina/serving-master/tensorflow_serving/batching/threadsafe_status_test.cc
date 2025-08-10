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

#include "machina_serving/batching/threadsafe_status.h"

#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/test.h"
#include "machina/core/protobuf/error_codes.pb.h"

namespace machina {
namespace serving {
namespace {

TEST(ThreadSafeStatus, DefaultOk) {
  ThreadSafeStatus status;
  TF_EXPECT_OK(status.status());
}

TEST(ThreadSafeStatus, Update) {
  ThreadSafeStatus status;
  TF_EXPECT_OK(status.status());

  status.Update(errors::FailedPrecondition("original error"));
  EXPECT_EQ(status.status().code(), error::FAILED_PRECONDITION);

  status.Update(absl::OkStatus());
  EXPECT_EQ(status.status().code(), error::FAILED_PRECONDITION);

  status.Update(errors::Internal("new error"));
  EXPECT_EQ(status.status().code(), error::FAILED_PRECONDITION);
}

TEST(ThreadSafeStatus, Move) {
  ThreadSafeStatus status;
  TF_EXPECT_OK(std::move(status).status());
}

}  // namespace
}  // namespace serving
}  // namespace machina
