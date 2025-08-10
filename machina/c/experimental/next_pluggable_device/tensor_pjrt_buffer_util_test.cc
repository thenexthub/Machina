/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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
#include "machina/c/experimental/next_pluggable_device/tensor_pjrt_buffer_util.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "machina/xla/pjrt/c/pjrt_c_api.h"
#include "machina/xla/pjrt/c/pjrt_c_api_cpu.h"
#include "machina/xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "machina/xla/pjrt/pjrt_api.h"
#include "machina/xla/pjrt/pjrt_c_api_client.h"
#include "machina/xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "machina/xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "machina/xla/shape.h"
#include "machina/xla/shape_util.h"
#include "machina/xla/tsl/lib/core/status_test_util.h"
#include "machina/xla/tsl/platform/status_matchers.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/xla/tsl/protobuf/error_codes.pb.h"
#include "machina/xla/xla_data.pb.h"
#include "machina/core/framework/types.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/tfrt/common/async_value_tensor.h"
#include "machina/core/tfrt/common/pjrt_util.h"
#include "tsl/platform/casts.h"

namespace machina {
namespace {

using ::testing::HasSubstr;
using ::testing::NotNull;
using ::tsl::testing::StatusIs;

PJRT_Buffer* CreateCBuffer() {
  auto status = pjrt::PjrtApi(DEVICE_CPU);
  if (!status.ok()) {
    CHECK_OK(pjrt::SetPjrtApi(DEVICE_CPU, GetPjrtApi()));
  }
  auto pjrt_client = xla::GetCApiClient(DEVICE_CPU);
  CHECK_OK(pjrt_client.status());
  auto c_api_client = down_cast<xla::PjRtCApiClient*>(pjrt_client->get());
  std::vector<int32_t> data(1, 0);
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::S32, {1});

  auto buffer = c_api_client->pjrt_c_client()->client->BufferFromHostBuffer(
      data.data(), shape.element_type(), shape.dimensions(),
      /*byte_strides=*/std::nullopt,
      xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
      c_api_client->pjrt_c_client()->client->memory_spaces()[0],
      /*device_layout=*/nullptr);
  CHECK_OK(buffer.status());

  return new PJRT_Buffer{std::move(*buffer), c_api_client->pjrt_c_client()};
}

TEST(TensorPjRtBufferUtilTest, GetPjRtCBufferFromTensorNoBuffer) {
  auto allocator = std::make_unique<AsyncValueAllocator>();
  machina::Tensor tensor(allocator.get(), DT_FLOAT, {1});

  EXPECT_THAT(
      GetPjRtCBufferFromTensor(&tensor),
      StatusIs(error::INTERNAL, HasSubstr(absl::StrCat(
                                    "Input tensor does not have PjRtBuffer"))));
}

TEST(TensorPjRtBufferUtilTest, GetPjRtCBufferFromTensorIncoorectType) {
  auto allocator = std::make_unique<AsyncValueAllocator>();
  machina::Tensor tensor(allocator.get(), DT_FLOAT, {1});
  xla::CpuClientOptions options;
  options.asynchronous = true;
  options.cpu_device_count = 1;

  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client, xla::GetXlaPjrtCpuClient(options));
  std::vector<int32_t> data(1, 0);
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::S32, {1});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      pjrt_client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          nullptr, pjrt_client->memory_spaces()[0], /*device_layout=*/nullptr));
  machina::AsyncValueTensor* av_tensor =
      machina::AsyncValueTensor::FromTensor(&tensor);
  av_tensor->SetBuffer(std::move(buffer));

  EXPECT_THAT(
      GetPjRtCBufferFromTensor(&tensor),
      StatusIs(
          error::INTERNAL,
          HasSubstr(absl::StrCat(
              "The PjRtBuffer in the tensor is not type PjRtCApiBuffer"))));
}

TEST(TensorPjRtBufferUtilTest, GetPjRtCBufferFromTensorSuccess) {
  auto allocator = std::make_unique<AsyncValueAllocator>();
  machina::Tensor tensor(allocator.get(), DT_FLOAT, {1});
  auto status = pjrt::PjrtApi(DEVICE_CPU);
  if (!status.ok()) {
    TF_ASSERT_OK(pjrt::SetPjrtApi(DEVICE_CPU, GetPjrtApi()));
  }
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client, xla::GetCApiClient(DEVICE_CPU));
  std::vector<int32_t> data(1, 0);
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::S32, {1});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      pjrt_client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          nullptr, pjrt_client->memory_spaces()[0], /*device_layout=*/nullptr));
  machina::AsyncValueTensor* av_tensor =
      machina::AsyncValueTensor::FromTensor(&tensor);
  av_tensor->SetBuffer(std::move(buffer));

  TF_ASSERT_OK_AND_ASSIGN(auto c_buffer, GetPjRtCBufferFromTensor(&tensor));

  EXPECT_THAT(c_buffer, NotNull());
}

TEST(TensorPjRtBufferUtilTest, SetPjRtCBufferToTensorNotAsyncValueTensor) {
  machina::Tensor tensor(DT_FLOAT, {1});
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client, xla::GetCApiClient(DEVICE_CPU));
  PJRT_Buffer* c_buffer = CreateCBuffer();

  TF_EXPECT_OK(SetPjRtCBufferToTensor(
      c_buffer, down_cast<xla::PjRtCApiClient*>(pjrt_client.get()), &tensor));
}

TEST(TensorPjRtBufferUtilTest, SetPjRtCBufferToTensorSuccess) {
  auto allocator = std::make_unique<AsyncValueAllocator>();
  machina::Tensor tensor(allocator.get(), DT_FLOAT, {1});
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client, xla::GetCApiClient(DEVICE_CPU));
  PJRT_Buffer* c_buffer = CreateCBuffer();

  TF_EXPECT_OK(SetPjRtCBufferToTensor(
      c_buffer, down_cast<xla::PjRtCApiClient*>(pjrt_client.get()), &tensor));
}

TEST(TensorPjRtBufferUtilTest, GetPjRtCApiClientNotFound) {
  EXPECT_THAT(
      GetPjRtCApiClient(machina::DeviceType(DEVICE_CPU)),
      StatusIs(error::NOT_FOUND,
               HasSubstr(absl::StrCat("PjRt client not found for device type ",
                                      DEVICE_CPU))));
}

TEST(TensorPjRtBufferUtilTest, GetPjRtCApiClientIncorrectType) {
  xla::CpuClientOptions options;
  options.asynchronous = true;
  options.cpu_device_count = 1;
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client, xla::GetXlaPjrtCpuClient(options));

  TF_ASSERT_OK(SetPjRtClientInTFGlobalResourceManager(DEVICE_CPU,
                                                      std::move(pjrt_client)));

  EXPECT_THAT(GetPjRtCApiClient(machina::DeviceType(DEVICE_CPU)),
              StatusIs(error::INTERNAL,
                       HasSubstr(absl::StrCat("PjRtClient for ", DEVICE_CPU,
                                              " is not type PjRtCApiClient"))));
}

TEST(TensorPjRtBufferUtilTest, GetPjRtCApiClientSuccess) {
  auto status = pjrt::PjrtApi(DEVICE_CPU);
  if (!status.ok()) {
    TF_ASSERT_OK(pjrt::SetPjrtApi(DEVICE_CPU, GetPjrtApi()));
  }
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client, xla::GetCApiClient(DEVICE_CPU));
  TF_ASSERT_OK(SetPjRtClientInTFGlobalResourceManager(DEVICE_CPU,
                                                      std::move(pjrt_client)));

  TF_ASSERT_OK_AND_ASSIGN(
      auto pjrt_client_get,
      GetPjRtCApiClient(machina::DeviceType(DEVICE_CPU)));

  EXPECT_THAT(pjrt_client_get, NotNull());
}

}  // namespace
}  // namespace machina
