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

#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "machina/compiler/jit/flags.h"
#include "machina/compiler/jit/mark_for_compilation_pass.h"
#include "machina/compiler/jit/tests/device_compiler_test_helper.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/lib/core/status_test_util.h"

namespace machina {
namespace {

TEST_F(DeviceCompilerSerializeTest, PersistentCacheOptionsTest) {
  GraphDef graph = GetTestGraph({-1, 4});

  // Warmup the persistent cache(s) with multiple runs. 4 is a magic number to
  // detect non-determinism in TF when running the test.
  listener()->ClearListenerHistory();
  for (int b = 1; b < 4; ++b) {
    TF_ASSERT_OK(ExecuteWithBatch(graph, b));
  }
  TF_ASSERT_OK(listener()->VerifyPersistentCacheUseListenerHistory(
      /*expect_persistent_cache_use=*/false));

  // Reset the cluster numbering between sessions so we can get the same
  // cluster numbering.
  testing::ResetClusterSequenceNumber();

  auto status =
      AlterPersistentCacheEntryHloModuleNames(machina::testing::TmpDir());
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(
      status.message(),
      "Did not find any persistent XLA compilation cache entries to alter."));

  TF_ASSERT_OK(AlterPersistentCacheEntryHloModuleNames(
      machina::testing::TmpDir(), "my_test_prefix"));

  // Run again and these should all hit in the persistent cache despite having
  // altered the persistent cache entries' HLO modules (disabled strict
  // signature checks).
  listener()->ClearListenerHistory();
  for (int b = 1; b < 4; ++b) {
    TF_ASSERT_OK(ExecuteWithBatch(graph, b));
  }
  TF_ASSERT_OK(listener()->VerifyPersistentCacheUseListenerHistory(
      /*expect_persistent_cache_use=*/true));
}

}  // namespace
}  // namespace machina

int main(int argc, char** argv) {
  machina::GetMarkForCompilationPassFlags()
      ->tf_xla_deterministic_cluster_names = true;
  machina::GetMarkForCompilationPassFlags()
      ->tf_xla_persistent_cache_directory = machina::testing::TmpDir();
  machina::GetMarkForCompilationPassFlags()
      ->tf_xla_disable_strict_signature_checks = true;
  machina::GetMarkForCompilationPassFlags()->tf_xla_persistent_cache_prefix =
      "my_test_prefix";
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
