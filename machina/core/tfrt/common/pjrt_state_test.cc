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
#include "machina/core/tfrt/common/pjrt_state.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/xla/pjrt/pjrt_client.h"
#include "machina/xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "machina/xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "machina/xla/tsl/lib/core/status_test_util.h"
#include "machina/xla/tsl/protobuf/error_codes.pb.h"
#include "machina/core/framework/types.h"
#include "machina/core/platform/refcount.h"
#include "machina/core/protobuf/error_codes.pb.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace {

using machina::PjRtState;
using ::testing::HasSubstr;
using ::tsl::testing::StatusIs;

class PjRtStateTestFixture : public testing::Test {
 protected:
  PjRtStateTestFixture() { pjrt_state_ = PjRtState::Create(); }
  ~PjRtStateTestFixture() override {
    machina::core::ScopedUnref pjrt_state_ref(pjrt_state_);
  }
  PjRtState* pjrt_state_;
};

TEST_F(PjRtStateTestFixture, SetAndGetPjRtClient) {
  xla::CpuClientOptions options;
  options.asynchronous = true;
  options.cpu_device_count = 1;
  TF_ASSERT_OK(pjrt_state_->SetPjRtClient(
      machina::DEVICE_CPU, xla::GetXlaPjrtCpuClient(options).value()));
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client,
                          pjrt_state_->GetPjRtClient(machina::DEVICE_CPU));
  EXPECT_THAT(pjrt_client, testing::NotNull());
}

TEST_F(PjRtStateTestFixture, AddAlreadyExistsPjRtClient) {
  xla::CpuClientOptions options;
  options.asynchronous = true;
  options.cpu_device_count = 1;

  TF_ASSERT_OK(pjrt_state_->SetPjRtClient(
      machina::DEVICE_CPU, xla::GetXlaPjrtCpuClient(options).value()));
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client_1,
                          pjrt_state_->GetPjRtClient(machina::DEVICE_CPU));

  TF_ASSERT_OK(pjrt_state_->SetPjRtClient(
      machina::DEVICE_CPU, xla::GetXlaPjrtCpuClient(options).value()));
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client_2,
                          pjrt_state_->GetPjRtClient(machina::DEVICE_CPU));

  EXPECT_NE(pjrt_client_1, pjrt_client_2);
}

TEST_F(PjRtStateTestFixture, GetNotExistPjRtClient) {
  EXPECT_THAT(pjrt_state_->GetPjRtClient(machina::DEVICE_CPU),
              absl_testing::StatusIs(
                  machina::error::NOT_FOUND,
                  HasSubstr("PjRt client not found for device type")));
}

TEST_F(PjRtStateTestFixture, DeletePjRtClient) {
  xla::CpuClientOptions options;
  options.asynchronous = true;
  options.cpu_device_count = 1;

  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client, xla::GetXlaPjrtCpuClient(options));
  xla::PjRtClient* pjrt_client_ptr = pjrt_client.get();
  TF_ASSERT_OK(pjrt_state_->SetPjRtClient(machina::DEVICE_CPU,
                                          std::move(pjrt_client)));

  TF_ASSERT_OK(pjrt_state_->MovePjRtClientToUnused(machina::DEVICE_CPU));

  EXPECT_THAT(pjrt_state_->GetPjRtClient(machina::DEVICE_CPU),
              absl_testing::StatusIs(
                  machina::error::NOT_FOUND,
                  HasSubstr("PjRt client not found for device type")));
  // Verifies that the PJRT client is still alive.
  EXPECT_EQ(pjrt_client_ptr->platform_name(), "cpu");
}

TEST_F(PjRtStateTestFixture, DeleteNotExistPjRtClient) {
  EXPECT_THAT(pjrt_state_->MovePjRtClientToUnused(machina::DEVICE_CPU),
              absl_testing::StatusIs(
                  machina::error::NOT_FOUND,
                  HasSubstr("PjRt client not found for device type")));
}

TEST_F(PjRtStateTestFixture, GetOrCreatePjRtClientExist) {
  xla::CpuClientOptions options;
  options.asynchronous = true;
  options.cpu_device_count = 1;

  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client, xla::GetXlaPjrtCpuClient(options));
  auto pjrt_client_ptr = pjrt_client.get();
  TF_ASSERT_OK(pjrt_state_->SetPjRtClient(machina::DEVICE_CPU,
                                          std::move(pjrt_client)));
  TF_ASSERT_OK_AND_ASSIGN(
      auto pjrt_client_get,
      pjrt_state_->GetOrCreatePjRtClient(machina::DEVICE_CPU));
  EXPECT_THAT(pjrt_client_get, pjrt_client_ptr);
}

TEST_F(PjRtStateTestFixture, GetOrCreatePjRtClientNotExist) {
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client, pjrt_state_->GetOrCreatePjRtClient(
                                                machina::DEVICE_CPU));
  EXPECT_THAT(pjrt_client, testing::NotNull());
}

}  // namespace
