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

#include "machina_serving/util/status_util.h"

#include <gtest/gtest.h>

namespace machina {
namespace serving {
namespace {

TEST(StatusUtilTest, ConvertsErrorStatusToStatusProto) {
  Status status =
      Status(static_cast<absl::StatusCode>(absl::StatusCode::kAborted),
             "aborted error message");
  StatusProto status_proto = ToStatusProto(status);
  EXPECT_EQ(machina::error::ABORTED, status_proto.error_code());
  EXPECT_EQ("aborted error message", status_proto.error_message());
}

TEST(StatusUtilTest, ConvertsOkStatusToStatusProto) {
  Status status;
  StatusProto status_proto = ToStatusProto(status);
  EXPECT_EQ(machina::error::OK, status_proto.error_code());
  EXPECT_EQ("", status_proto.error_message());
}

TEST(StatusUtilTest, ConvertsErrorStatusProtoToStatus) {
  StatusProto status_proto;
  status_proto.set_error_code(machina::error::ALREADY_EXISTS);
  status_proto.set_error_message("already exists error message");
  Status status = FromStatusProto(status_proto);
  EXPECT_EQ(machina::error::ALREADY_EXISTS, status.code());
  EXPECT_EQ("already exists error message", status.message());
}

TEST(StatusUtilTest, ConvertsOkStatusProtoToStatus) {
  StatusProto status_proto;
  status_proto.set_error_code(machina::error::OK);
  Status status = FromStatusProto(status_proto);
  EXPECT_EQ(machina::error::OK, status.code());
  EXPECT_EQ("", status.message());
}

}  // namespace

}  // namespace serving
}  // namespace machina
