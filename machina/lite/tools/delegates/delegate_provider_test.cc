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
#include "machina/lite/tools/delegates/delegate_provider.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "machina/lite/c/test_util.h"
#include "machina/lite/tools/tool_params.h"

namespace tflite {
namespace tools {
namespace {
TEST(ProvidedDelegateListTest, AddAllDelegateParams) {
  ToolParams params;
  ProvidedDelegateList providers(&params);
  providers.AddAllDelegateParams();
  // As we link the test with XNNPACK and nnapi delegate providers, we should
  // expect to have these two knob parameters.
  EXPECT_TRUE(params.HasParam("use_xnnpack"));

// TODO(b/249485631): Enable the check after NNAPI Delegate Provider supports
// stable TFLite ABI.
#if !TFLITE_WITH_STABLE_ABI
  EXPECT_TRUE(params.HasParam("use_nnapi"));
#endif  // !TFLITE_WITH_STABLE_ABI
}

TEST(ProvidedDelegateListTest, AppendCmdlineFlags) {
  std::vector<Flag> flags;

  ToolParams params;
  ProvidedDelegateList providers(&params);
  providers.AddAllDelegateParams();

  providers.AppendCmdlineFlags(flags);
  EXPECT_FALSE(flags.empty());
}

TEST(KernelTestDelegateProvidersTest, CreateAllRankedDelegates) {
#if !defined(__Fuchsia__) && !defined(__s390x__) && \
    !defined(TFLITE_WITHOUT_XNNPACK)
  ToolParams params;
  ProvidedDelegateList providers(&params);
  providers.AddAllDelegateParams();

// TODO(b/249054271): Dummy delegate hasn't been migrated to use TFLite with
// stable ABI yet. The check here can be removed after the extension.
#if TFLITE_WITH_STABLE_ABI
  ASSERT_EQ(TfLiteInitializeShimsForTest(), 0);
  params.Set<bool>("use_xnnpack", true, 1);

  auto delegates = providers.CreateAllRankedDelegates();
  EXPECT_EQ(1, delegates.size());

  EXPECT_EQ("XNNPACK", delegates.front().provider->GetName());
  EXPECT_NE(nullptr, delegates.front().delegate.get());
  EXPECT_EQ(1, delegates.front().rank);
#else   // TFLITE_WITH_STABLE_ABI
  // We set the position of "use_xnnpack" to be smaller than that of
  // "use_dummy_delegate" so that the Dummy delegate will be ahead of the
  // XNNPACK delegate in the returned list.
  params.Set<bool>("use_xnnpack", true, 2);
  params.Set<bool>("use_dummy_delegate", true, 1);

  auto delegates = providers.CreateAllRankedDelegates();
  EXPECT_EQ(2, delegates.size());

  EXPECT_EQ("DummyDelegate", delegates.front().provider->GetName());
  EXPECT_EQ(1, delegates.front().rank);
  EXPECT_NE(nullptr, delegates.front().delegate.get());

  EXPECT_EQ("XNNPACK", delegates.back().provider->GetName());
  EXPECT_NE(nullptr, delegates.back().delegate.get());
  EXPECT_EQ(2, delegates.back().rank);
#endif  // TFLITE_WITH_STABLE_ABI
#endif
}
}  // namespace
}  // namespace tools
}  // namespace tflite
