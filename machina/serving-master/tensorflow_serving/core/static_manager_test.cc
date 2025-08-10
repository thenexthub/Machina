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

#include "machina_serving/core/static_manager.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace machina {
namespace serving {
namespace {

TEST(StaticManagerTest, StoresServables) {
  StaticManagerBuilder builder;
  TF_CHECK_OK(builder.AddServable(ServableId{"name", 22},
                                  std::unique_ptr<int>{new int{22}}));
  auto manager = builder.Build();
  ServableHandle<int> handle;
  TF_CHECK_OK(manager->GetServableHandle(ServableRequest::Specific("name", 22),
                                         &handle));
  EXPECT_EQ(22, *handle);
}

TEST(StaticManagerTest, UseAfterBuild) {
  StaticManagerBuilder builder;
  TF_CHECK_OK(builder.AddServable(ServableId{"name", 22},
                                  std::unique_ptr<int>{new int{22}}));
  auto manager = builder.Build();
  EXPECT_EQ(nullptr, builder.Build());
  EXPECT_FALSE(builder
                   .AddServable(ServableId{"name", 21},
                                std::unique_ptr<int>(new int(22)))
                   .ok());
}

TEST(StaticManagerTest, Errors) {
  StaticManagerBuilder builder;
  // Null servable.
  EXPECT_FALSE(
      builder.AddServable(ServableId{"name", 22}, std::unique_ptr<int>{nullptr})
          .ok());
  // Double add.
  TF_CHECK_OK(builder.AddServable(ServableId{"name", 22},
                                  std::unique_ptr<int>{new int{22}}));
  EXPECT_FALSE(builder
                   .AddServable(ServableId{"name", 22},
                                std::unique_ptr<int>{new int{22}})
                   .ok());
}

TEST(StaticManagerTest, GetLatestVersion) {
  StaticManagerBuilder builder;
  TF_CHECK_OK(builder.AddServable(ServableId{"name", 22},
                                  std::unique_ptr<int>{new int{22}}));
  const ServableId id = {"name", 24};
  TF_CHECK_OK(builder.AddServable(id, std::unique_ptr<int>{new int{24}}));
  auto manager = builder.Build();

  ServableHandle<int> handle;
  TF_CHECK_OK(
      manager->GetServableHandle(ServableRequest::Latest("name"), &handle));
  EXPECT_EQ(24, *handle);
  EXPECT_EQ(id, handle.id());
}

TEST(StaticManagerTest, GetSpecificVersion) {
  StaticManagerBuilder builder;
  const ServableId id = {"name", 22};
  TF_CHECK_OK(builder.AddServable(id, std::unique_ptr<int>{new int{22}}));
  TF_CHECK_OK(builder.AddServable(ServableId{"name", 24},
                                  std::unique_ptr<int>{new int{24}}));
  auto manager = builder.Build();

  ServableHandle<int> handle;
  TF_CHECK_OK(manager->GetServableHandle(ServableRequest::FromId(id), &handle));
  EXPECT_EQ(22, *handle);
  EXPECT_EQ(id, handle.id());
}

TEST(StaticManagerTest, ServableNotFound) {
  StaticManagerBuilder builder;
  auto manager = builder.Build();
  ServableHandle<int> handle;
  EXPECT_EQ(error::NOT_FOUND,
            manager->GetServableHandle(ServableRequest::Latest("name"), &handle)
                .code());
  EXPECT_EQ(nullptr, handle.get());
}

TEST(StaticManagerTest, VersionNotFound) {
  StaticManagerBuilder builder;
  TF_CHECK_OK(builder.AddServable(ServableId{"name", 22},
                                  std::unique_ptr<int>{new int{22}}));
  auto manager = builder.Build();
  ServableHandle<int> handle;
  EXPECT_EQ(
      error::NOT_FOUND,
      manager->GetServableHandle(ServableRequest::Specific("name", 21), &handle)
          .code());
  EXPECT_EQ(nullptr, handle.get());
}

}  // namespace
}  // namespace serving
}  // namespace machina
