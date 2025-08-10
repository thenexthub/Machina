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
#include "machina/core/tfrt/common/create_pjrt_client_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"  // IWYU pragma: keep
#include "machina/xla/tsl/protobuf/error_codes.pb.h"
#include "machina/core/framework/types.h"
#include "tsl/platform/status_matchers.h"

namespace machina {
namespace {

using ::testing::HasSubstr;
using ::tsl::testing::StatusIs;

TEST(CreatePjRtClientTest, GetNotExistPjRtClientNotImplemented) {
  EXPECT_THAT(GetOrCreatePjRtClient(DEVICE_CPU),
              absl_testing::StatusIs(
                  error::NOT_FOUND,
                  HasSubstr(absl::StrCat("The PJRT client factory of `",
                                         DEVICE_CPU, "` is not registered"))));
}

#if GOOGLE_CUDA || MACHINA_USE_ROCM
TEST(CreatePjRtClientTest, GetNotExistGpuPjRtClient) {
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client,
                          GetOrCreatePjRtClient(DEVICE_MACHINA_XLAGPU));
  EXPECT_THAT(pjrt_client, ::testing::NotNull());
}
#endif

}  // namespace
}  // namespace machina
