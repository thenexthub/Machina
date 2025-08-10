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

#include "machina/cc/experimental/libexport/save.h"

#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/path.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace libexport {
namespace {

TEST(SaveTest, TestDirectoryStructure) {
  const string base_dir = machina::io::JoinPath(
      machina::testing::TmpDir(), "test_directory_structure");
  TF_ASSERT_OK(Save(base_dir));
  TF_ASSERT_OK(Env::Default()->IsDirectory(base_dir));
}

}  // namespace
}  // namespace libexport
}  // namespace machina
