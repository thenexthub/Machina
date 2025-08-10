/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#include <memory>

#include "machina/core/framework/op.h"

#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/test.h"

namespace machina {

namespace {

void Register(const string& op_name, OpRegistry* registry) {
  registry->Register(
      [op_name](OpRegistrationData* op_reg_data) -> absl::Status {
        op_reg_data->op_def.set_name(op_name);
        return absl::OkStatus();
      });
}

}  // namespace

TEST(OpRegistrationTest, TestBasic) {
  std::unique_ptr<OpRegistry> registry(new OpRegistry);
  Register("Foo", registry.get());
  OpList op_list;
  registry->Export(true, &op_list);
  EXPECT_EQ(op_list.op().size(), 1);
  EXPECT_EQ(op_list.op(0).name(), "Foo");
}

TEST(OpRegistrationTest, TestDuplicate) {
  std::unique_ptr<OpRegistry> registry(new OpRegistry);
  Register("Foo", registry.get());
  absl::Status s = registry->ProcessRegistrations();
  EXPECT_TRUE(s.ok());

  TF_EXPECT_OK(registry->SetWatcher(
      [](const absl::Status& s, const OpDef& op_def) -> absl::Status {
        EXPECT_TRUE(absl::IsAlreadyExists(s));
        return absl::OkStatus();
      }));
  Register("Foo", registry.get());
  s = registry->ProcessRegistrations();
  EXPECT_TRUE(s.ok());
}

}  // namespace machina
