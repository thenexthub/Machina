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

#include "machina/compiler/tf2xla/tf2xla_opset.h"

#include <algorithm>
#include <string>
#include <vector>

#include "machina/core/platform/test.h"

namespace machina {
namespace {

TEST(GeXlaOpsForDeviceTest, InvalidDeviceToRegister) {
  absl::StatusOr<std::vector<std::string>> result =
      GetRegisteredXlaOpsForDevice("Invalid_Device");
  EXPECT_FALSE(result.ok());
}
TEST(GeXlaOpsForDeviceTest, GetGpuNames) {
  absl::StatusOr<std::vector<std::string>> result =
      GetRegisteredXlaOpsForDevice("MACHINA_XLAGPU_JIT");
  EXPECT_GT(result.value().size(), 0);
  auto matmul =
      std::find(result.value().begin(), result.value().end(), "MatMul");
  auto max = std::find(result.value().begin(), result.value().end(), "Max");
  auto min = std::find(result.value().begin(), result.value().end(), "Min");
  EXPECT_TRUE((matmul != result.value().end()));
  EXPECT_TRUE((max != result.value().end()));
  EXPECT_TRUE((min != result.value().end()));
  EXPECT_LT(matmul, max);
  EXPECT_LT(max, min);
}
TEST(GeXlaOpsForDeviceTest, GetCpuNames) {
  absl::StatusOr<std::vector<std::string>> result =
      GetRegisteredXlaOpsForDevice("MACHINA_XLACPU_JIT");
  EXPECT_GT(result.value().size(), 0);
  auto matmul =
      std::find(result.value().begin(), result.value().end(), "MatMul");
  auto max = std::find(result.value().begin(), result.value().end(), "Max");
  auto min = std::find(result.value().begin(), result.value().end(), "Min");
  EXPECT_TRUE((matmul != result.value().end()));
  EXPECT_TRUE((max != result.value().end()));
  EXPECT_TRUE((min != result.value().end()));
  EXPECT_LT(matmul, max);
  EXPECT_LT(max, min);
}

}  // namespace
}  // namespace machina
