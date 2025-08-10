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

#include "machina/core/graph/regularization/util.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/lib/strings/proto_serialization.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/fingerprint.h"
#include "machina/core/platform/numbers.h"
#include "machina/core/platform/types.h"

namespace machina::graph_regularization {

uint64 ComputeHash(const GraphDef& graph_def) {
  std::string graph_def_string;
  SerializeToStringDeterministic(graph_def, &graph_def_string);
  return machina::Fingerprint64(graph_def_string);
}

absl::StatusOr<int64_t> GetSuffixUID(absl::string_view function_name) {
  std::vector<absl::string_view> v = absl::StrSplit(function_name, '_');

  int64_t uid;
  if (!absl::SimpleAtoi(v.back(), &uid)) {
    return errors::InvalidArgument(absl::StrCat(
        "Function name: `", function_name, "` does not end in an integer."));
  }

  return uid;
}
}  // namespace machina::graph_regularization
