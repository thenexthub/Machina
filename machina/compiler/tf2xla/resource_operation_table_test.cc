/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#include "machina/compiler/tf2xla/resource_operation_table.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_join.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace {
bool IsResourceArgDef(const OpDef::ArgDef& arg_def) {
  return arg_def.type() == DT_RESOURCE;
}

bool HasResourceInputOrOutput(const OpDef& op_def) {
  return absl::c_any_of(op_def.input_arg(), IsResourceArgDef) ||
         absl::c_any_of(op_def.output_arg(), IsResourceArgDef);
}

TEST(ResourceOperationTableTest, HaveAllResourceOps) {
  absl::flat_hash_map<string, bool> known_resource_ops;
  for (absl::string_view known_resource_op :
       resource_op_table_internal::GetKnownResourceOps()) {
    ASSERT_TRUE(
        known_resource_ops.insert({string(known_resource_op), false}).second);
  }

  std::vector<string> xla_op_names = XlaOpRegistry::GetAllRegisteredOps();
  for (const string& xla_op_name : xla_op_names) {
    const OpDef* op_def;
    TF_ASSERT_OK(OpRegistry::Global()->LookUpOpDef(xla_op_name, &op_def));
    if (HasResourceInputOrOutput(*op_def)) {
      EXPECT_EQ(known_resource_ops.count(xla_op_name), 1)
          << "Unknown resource op " << xla_op_name;
      known_resource_ops[xla_op_name] = true;
    }
  }

  std::vector<string> unnecessary_resource_ops;
  for (const auto& pair : known_resource_ops) {
    if (!pair.second) {
      unnecessary_resource_ops.push_back(pair.first);
    }
  }

  EXPECT_TRUE(unnecessary_resource_ops.empty())
      << "Stale resource ops:\n"
      << absl::StrJoin(unnecessary_resource_ops, "\n");
}
}  // namespace
}  // namespace machina
