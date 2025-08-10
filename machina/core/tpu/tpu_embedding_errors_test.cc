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

#include "machina/core/tpu/tpu_embedding_errors.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "machina/xla/tsl/lib/core/status_test_util.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/xla/tsl/protobuf/error_codes.pb.h"
#include "machina/core/platform/errors.h"

namespace machina::tpu {
namespace {

using absl::Status;
using absl::StatusOr;

StatusOr<std::string> GenerateTFStatusOr(absl::StatusCode code,
                                         absl::string_view value = "") {
  if (code == absl::StatusCode::kOk) {
    return std::string(value);
  } else {
    return absl::Status(code, value);
  }
}

TEST(TpuEmbeddingErrors, StatusOk) {
  constexpr absl::string_view kValue = "success";

  {
    const Status status = AppendTpuEmbeddingErrorPayload(absl::OkStatus());
    TF_EXPECT_OK(status);
    EXPECT_FALSE(HasTpuEmbeddingErrorPayload(status));
    EXPECT_FALSE(HasTpuEmbeddingErrorMessage(status));
  }

  {
    TF_ASSERT_OK_AND_ASSIGN(const std::string value,
                            AppendTpuEmbeddingErrorPayload(GenerateTFStatusOr(
                                absl::StatusCode::kOk, kValue)));
    EXPECT_EQ(value, kValue);
  }
}

TEST(TpuEmbeddingErrors, StatusFailed) {
  {
    const Status status =
        AppendTpuEmbeddingErrorPayload(errors::InvalidArgument(""));
    EXPECT_EQ(status.code(), error::Code::INVALID_ARGUMENT);
    EXPECT_TRUE(HasTpuEmbeddingErrorPayload(status));
    EXPECT_TRUE(HasTpuEmbeddingErrorMessage(status));
  }

  {
    StatusOr<std::string> status_or = AppendTpuEmbeddingErrorPayload(
        GenerateTFStatusOr(absl::StatusCode::kResourceExhausted));
    EXPECT_FALSE(status_or.ok());
    const Status& status = status_or.status();
    EXPECT_EQ(status.code(), error::Code::RESOURCE_EXHAUSTED);
    EXPECT_TRUE(HasTpuEmbeddingErrorPayload(status));
    EXPECT_TRUE(HasTpuEmbeddingErrorMessage(status));
  }
}

}  // namespace
}  // namespace machina::tpu
