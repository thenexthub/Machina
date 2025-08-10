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

#include "machina/core/common_runtime/request_cost_accessor_registry.h"

#include "absl/time/time.h"
#include "machina/core/common_runtime/request_cost_accessor.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace {

class TestRequestCostAccessor : public RequestCostAccessor {
 public:
  RequestCost* GetRequestCost() const override { return nullptr; }
};

REGISTER_REQUEST_COST_ACCESSOR("test", TestRequestCostAccessor);

TEST(RequestCostAccessorRegistryTest, Basic) {
  std::unique_ptr<const RequestCostAccessor> test_accessor =
      RequestCostAccessorRegistry::CreateByNameOrNull("unregistered");
  EXPECT_EQ(test_accessor, nullptr);

  test_accessor = RequestCostAccessorRegistry::CreateByNameOrNull("test");
  EXPECT_NE(test_accessor, nullptr);
}

TEST(RequestCostAccessorRegistryDeathTest, CrashWhenRegisterTwice) {
  const auto creator = []() {
    return std::make_unique<TestRequestCostAccessor>();
  };
  EXPECT_DEATH(
      RequestCostAccessorRegistry::RegisterRequestCostAccessor("test", creator),
      "RequestCostAccessor test is registered twice.");
}

}  // namespace
}  // namespace machina
