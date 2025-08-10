/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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

#include "machina_serving/core/static_source_router.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/core/lib/core/status.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/lib/core/stringpiece.h"
#include "machina/core/platform/types.h"
#include "machina_serving/core/servable_data.h"
#include "machina_serving/core/source.h"
#include "machina_serving/core/storage_path.h"
#include "machina_serving/core/target.h"
#include "machina_serving/core/test_util/mock_storage_path_target.h"

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::StrictMock;

namespace machina {
namespace serving {
namespace {

TEST(StaticSourceRouterTest, Basic) {
  const std::vector<string> regexps = {"0th", "1st"};
  std::unique_ptr<StaticSourceRouter<StoragePath>> router;
  TF_ASSERT_OK(StaticSourceRouter<string>::Create(regexps, &router));
  std::vector<Source<StoragePath>*> output_ports = router->GetOutputPorts();
  ASSERT_EQ(regexps.size() + 1, output_ports.size());
  std::vector<std::unique_ptr<test_util::MockStoragePathTarget>> targets;
  for (int i = 0; i < output_ports.size(); ++i) {
    std::unique_ptr<test_util::MockStoragePathTarget> target(
        new StrictMock<test_util::MockStoragePathTarget>);
    ConnectSourceToTarget(output_ports[i], target.get());
    targets.push_back(std::move(target));
  }

  // Matches the 0th regexp. Should go to output port 0.
  EXPECT_CALL(
      *targets[0],
      SetAspiredVersions(Eq("0th_foo"), ElementsAre(ServableData<StoragePath>(
                                            {"0th_foo", 7}, "data"))));
  router->SetAspiredVersions(
      "0th_foo", {ServableData<StoragePath>({"0th_foo", 7}, "data")});

  // Matches the 0th and 1st regexps. Should go to output port 0.
  EXPECT_CALL(*targets[0],
              SetAspiredVersions(Eq("0th_foo_1st"),
                                 ElementsAre(ServableData<StoragePath>(
                                     {"0th_foo_1st", 7}, "data"))));
  router->SetAspiredVersions(
      "0th_foo_1st", {ServableData<StoragePath>({"0th_foo_1st", 7}, "data")});

  // Matches the 1st regexp but not the 0th. Should go to output port 1.
  EXPECT_CALL(
      *targets[1],
      SetAspiredVersions(Eq("foo_1st"), ElementsAre(ServableData<StoragePath>(
                                            {"foo_1st", 7}, "data"))));
  router->SetAspiredVersions(
      "foo_1st", {ServableData<StoragePath>({"foo_1st", 7}, "data")});

  // Doesn't match any of the regexps. Should go to output port 2.
  EXPECT_CALL(
      *targets[2],
      SetAspiredVersions(Eq("no_match"), ElementsAre(ServableData<StoragePath>(
                                             {"no_match", 7}, "data"))));
  router->SetAspiredVersions(
      "no_match", {ServableData<StoragePath>({"no_match", 7}, "data")});
}

}  // namespace
}  // namespace serving
}  // namespace machina
