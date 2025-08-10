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
#include "machina/core/tfrt/utils/utils.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/core/framework/device.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/statusor.h"
#include "tfrt/cpp_tests/test_util.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime

namespace tfrt {
namespace {

using ::testing::HasSubstr;
using ::testing::SizeIs;

TEST(UtilsTest, ConvertTfDTypeToTfrtDType) {
#define DTYPE(TFRT_DTYPE, TF_DTYPE)                          \
  EXPECT_EQ(ConvertTfDTypeToTfrtDType(machina::TF_DTYPE), \
            DType(DType::TFRT_DTYPE));
#include "machina/core/tfrt/utils/dtype.def"  // NOLINT

  EXPECT_EQ(ConvertTfDTypeToTfrtDType(machina::DT_HALF_REF), DType());
}

TEST(UtilsTest, CreateDummyTfDevices) {
  const std::vector<std::string> device_name{"/device:cpu:0", "/device:gpu:1"};
  std::vector<std::unique_ptr<machina::Device>> dummy_tf_devices;

  CreateDummyTfDevices(device_name, &dummy_tf_devices);

  ASSERT_THAT(dummy_tf_devices, SizeIs(2));

  EXPECT_EQ(dummy_tf_devices[0]->name(), device_name[0]);
  EXPECT_EQ(dummy_tf_devices[0]->device_type(), machina::DEVICE_TPU_SYSTEM);
  EXPECT_THAT(dummy_tf_devices[0]->attributes().physical_device_desc(),
              HasSubstr("device: TFRT TPU SYSTEM device"));
  EXPECT_EQ(dummy_tf_devices[1]->name(), device_name[1]);
}

TEST(UtilsTest, ReturnIfErrorInImport) {
  auto status = []() {
    RETURN_IF_ERROR_IN_IMPORT(
        machina::errors::CancelledWithPayloads("msg", {{"a", "b"}}));
    return absl::OkStatus();
  }();
  EXPECT_FALSE(status.ok());
  EXPECT_STREQ(status.ToString().c_str(),
               "CANCELLED: GraphDef proto -> MLIR: msg [a='b']");
  EXPECT_EQ(status.GetPayload("a"), "b");
}

TEST(UtilsTest, ReturnIfErrorInCompile) {
  auto status = []() {
    RETURN_IF_ERROR_IN_COMPILE(
        machina::errors::CancelledWithPayloads("msg", {{"a", "b"}}));
    return absl::OkStatus();
  }();
  EXPECT_FALSE(status.ok());
  EXPECT_STREQ(
      status.ToString().c_str(),
      "CANCELLED: TF dialect -> TFRT dialect, compiler issue, please contact "
      "the TFRT team: msg [a='b']");
  EXPECT_EQ(status.GetPayload("a"), "b");
}

TEST(UtilsTest, ReturnIfErrorInInit) {
  auto status = []() {
    RETURN_IF_ERROR_IN_INIT(
        machina::errors::CancelledWithPayloads("msg", {{"a", "b"}}));
    return absl::OkStatus();
  }();
  EXPECT_FALSE(status.ok());
  EXPECT_STREQ(status.ToString().c_str(),
               "CANCELLED: Initialize TFRT: msg [a='b']");
  EXPECT_EQ(status.GetPayload("a"), "b");
}

TEST(UtilsTest, AssignOrReturnInImport) {
  auto status = []() {
    ASSIGN_OR_RETURN_IN_IMPORT(
        [[maybe_unused]] auto unused_value,
        absl::StatusOr<int>(
            machina::errors::CancelledWithPayloads("msg", {{"a", "b"}})));
    return absl::OkStatus();
  }();
  EXPECT_FALSE(status.ok());
  EXPECT_STREQ(status.ToString().c_str(),
               "CANCELLED: GraphDef proto -> MLIR: msg [a='b']");
  EXPECT_EQ(status.GetPayload("a"), "b");
}

TEST(UtilsTest, AssignOrReturnInCompile) {
  auto status = []() {
    ASSIGN_OR_RETURN_IN_COMPILE(
        [[maybe_unused]] auto unused_value,
        absl::StatusOr<int>(
            machina::errors::CancelledWithPayloads("msg", {{"a", "b"}})));
    return absl::OkStatus();
  }();
  EXPECT_FALSE(status.ok());
  EXPECT_STREQ(status.ToString().c_str(),
               "CANCELLED: TF dialect -> TFRT dialect, compiler issue, please "
               "contact the TFRT team: msg [a='b']");
  EXPECT_EQ(status.GetPayload("a"), "b");
}

TEST(UtilsTest, AssignOrReturnInInit) {
  auto status = []() {
    ASSIGN_OR_RETURN_IN_INIT(
        [[maybe_unused]] auto unused_value,
        absl::StatusOr<int>(
            machina::errors::CancelledWithPayloads("msg", {{"a", "b"}})));
    return absl::OkStatus();
  }();
  EXPECT_FALSE(status.ok());
  EXPECT_STREQ(std::string(status.ToString()).c_str(),
               "CANCELLED: Initialize TFRT: msg [a='b']");
  EXPECT_EQ(status.GetPayload("a"), "b");
}

}  // namespace
}  // namespace tfrt
