/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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
#include "machina/core/data/service/utils.h"

#include <string>

#include "machina/core/data/service/common.pb.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/path.h"
#include "machina/core/platform/test.h"
#include "machina/core/protobuf/error_codes.pb.h"

namespace machina {
namespace data {

namespace {
DatasetDef DatasetDefWithVersion(int32_t version) {
  DatasetDef def;
  def.mutable_graph()->set_version(version);
  return def;
}
}  // namespace

TEST(Utils, ReadWriteDataset) {
  std::string filename = testing::TmpDir();
  ASSERT_TRUE(Env::Default()->CreateUniqueFileName(&filename, "journal_dir"));
  int32_t version = 3;
  DatasetDef def = DatasetDefWithVersion(version);
  TF_ASSERT_OK(WriteDatasetDef(filename, def));
  DatasetDef result;
  TF_ASSERT_OK(ReadDatasetDef(filename, result));
  EXPECT_EQ(result.graph().version(), version);
}

TEST(Utils, OverwriteDataset) {
  std::string filename = testing::TmpDir();
  ASSERT_TRUE(Env::Default()->CreateUniqueFileName(&filename, "journal_dir"));
  int32_t version_1 = 1;
  int32_t version_2 = 2;
  DatasetDef def_1 = DatasetDefWithVersion(version_1);
  TF_ASSERT_OK(WriteDatasetDef(filename, def_1));
  DatasetDef def_2 = DatasetDefWithVersion(version_2);
  TF_ASSERT_OK(WriteDatasetDef(filename, def_2));
  DatasetDef result;
  TF_ASSERT_OK(ReadDatasetDef(filename, result));
  EXPECT_EQ(result.graph().version(), version_2);
}

TEST(Utils, ReadDatasetNotFound) {
  std::string filename = testing::TmpDir();
  ASSERT_TRUE(Env::Default()->CreateUniqueFileName(&filename, "journal_dir"));
  DatasetDef result;
  absl::Status s = ReadDatasetDef(filename, result);
  EXPECT_EQ(s.code(), error::NOT_FOUND);
}

}  // namespace data
}  // namespace machina
