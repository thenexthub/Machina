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

#include "machina/core/common_runtime/build_graph_options.h"

#include "machina/core/lib/strings/strcat.h"

namespace machina {

string BuildGraphOptions::DebugString() const {
  string rv = "Feed endpoints: ";
  for (auto& s : callable_options.feed()) {
    strings::StrAppend(&rv, s, ", ");
  }
  strings::StrAppend(&rv, "\nFetch endpoints: ");
  for (auto& s : callable_options.fetch()) {
    strings::StrAppend(&rv, s, ", ");
  }
  strings::StrAppend(&rv, "\nTarget nodes: ");
  for (auto& s : callable_options.target()) {
    strings::StrAppend(&rv, s, ", ");
  }
  if (collective_graph_key != kNoCollectiveGraphKey) {
    strings::StrAppend(&rv, "\ncollective_graph_key: ", collective_graph_key);
  }
  string collective_order_str;
  switch (collective_order) {
    case GraphCollectiveOrder::kNone:
      collective_order_str = "none";
      break;
    case GraphCollectiveOrder::kEdges:
      collective_order_str = "edges";
      break;
    case GraphCollectiveOrder::kAttrs:
      collective_order_str = "attrs";
      break;
  }
  strings::StrAppend(&rv, "\ncollective_order: ", collective_order_str);
  return rv;
}

}  // namespace machina
