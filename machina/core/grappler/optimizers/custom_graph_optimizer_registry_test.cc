/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

#include "machina/core/grappler/optimizers/custom_graph_optimizer_registry.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "machina/core/grappler/optimizers/custom_graph_optimizer.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace grappler {
namespace {

static const char* kTestOptimizerName = "Test";
static const char* kTestPluginOptimizerName = "TestPlugin";

class TestGraphOptimizer : public CustomGraphOptimizer {
 public:
  absl::Status Init(
      const machina::RewriterConfig_CustomGraphOptimizer* config) override {
    return absl::OkStatus();
  }
  string name() const override { return kTestOptimizerName; }
  bool UsesFunctionLibrary() const override { return false; }
  absl::Status Optimize(Cluster* cluster, const GrapplerItem& item,
                        GraphDef* optimized_graph) override {
    return absl::OkStatus();
  }
};

REGISTER_GRAPH_OPTIMIZER_AS(TestGraphOptimizer, "StaticRegister");

TEST(CustomGraphOptimizerRegistryTest, DynamicRegistration) {
  std::vector<string> optimizers =
      CustomGraphOptimizerRegistry::GetRegisteredOptimizers();
  std::unique_ptr<const CustomGraphOptimizer> test_optimizer;
  ASSERT_EQ(
      0, std::count(optimizers.begin(), optimizers.end(), "DynamicRegister"));
  test_optimizer =
      CustomGraphOptimizerRegistry::CreateByNameOrNull("DynamicRegister");
  EXPECT_EQ(nullptr, test_optimizer);
  CustomGraphOptimizerRegistry::RegisterOptimizerOrDie(
      []() { return new TestGraphOptimizer; }, "DynamicRegister");
  optimizers = CustomGraphOptimizerRegistry::GetRegisteredOptimizers();
  ASSERT_EQ(
      1, std::count(optimizers.begin(), optimizers.end(), "DynamicRegister"));
  test_optimizer =
      CustomGraphOptimizerRegistry::CreateByNameOrNull("DynamicRegister");
  ASSERT_NE(nullptr, test_optimizer);
  EXPECT_EQ(kTestOptimizerName, test_optimizer->name());
}

TEST(CustomGraphOptimizerRegistryTest, StaticRegistration) {
  const std::vector<string> optimizers =
      CustomGraphOptimizerRegistry::GetRegisteredOptimizers();
  EXPECT_EQ(1,
            std::count(optimizers.begin(), optimizers.end(), "StaticRegister"));
  std::unique_ptr<const CustomGraphOptimizer> test_optimizer =
      CustomGraphOptimizerRegistry::CreateByNameOrNull("StaticRegister");
  ASSERT_NE(nullptr, test_optimizer);
  EXPECT_EQ(kTestOptimizerName, test_optimizer->name());
}

TEST(GraphOptimizerRegistryTest, CrashesOnDuplicateRegistration) {
  const auto creator = []() { return new TestGraphOptimizer; };
  EXPECT_DEATH(CustomGraphOptimizerRegistry::RegisterOptimizerOrDie(
                   creator, "StaticRegister"),
               "twice");
}

class TestPluginGraphOptimizer : public CustomGraphOptimizer {
 public:
  absl::Status Init(
      const machina::RewriterConfig_CustomGraphOptimizer* config) override {
    return absl::OkStatus();
  }
  string name() const override { return kTestPluginOptimizerName; }
  bool UsesFunctionLibrary() const override { return false; }
  absl::Status Optimize(Cluster* cluster, const GrapplerItem& item,
                        GraphDef* optimized_graph) override {
    return absl::OkStatus();
  }
};

TEST(PluginGraphOptimizerRegistryTest, CrashesOnDuplicateRegistration) {
  const auto creator = []() { return new TestPluginGraphOptimizer; };
  ConfigList config_list;
  PluginGraphOptimizerRegistry::RegisterPluginOptimizerOrDie(creator, "GPU",
                                                             config_list);
  PluginGraphOptimizerRegistry::RegisterPluginOptimizerOrDie(creator, "CPU",
                                                             config_list);
  EXPECT_DEATH(PluginGraphOptimizerRegistry::RegisterPluginOptimizerOrDie(
                   creator, "GPU", config_list),
               "twice");
}

}  // namespace
}  // namespace grappler
}  // namespace machina
