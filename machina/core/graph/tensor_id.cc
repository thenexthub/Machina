/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#include "machina/core/graph/tensor_id.h"

#include <string>

#include "machina/core/lib/core/stringpiece.h"
#include "machina/core/lib/strings/str_util.h"

namespace machina {

TensorId::TensorId(const SafeTensorId& id) : TensorId(id.first, id.second) {}

SafeTensorId::SafeTensorId(const TensorId& id)
    : SafeTensorId(string(id.first), id.second) {}

TensorId ParseTensorName(const string& name) {
  return ParseTensorName(absl::string_view(name.data(), name.size()));
}

TensorId ParseTensorName(absl::string_view name) {
  // Parse either a name, ^name, or name:digits.  To do so, we go backwards from
  // the end of the string, skipping over a run of digits.  If we hit a ':'
  // character, then we know we are in the 'name:digits' regime.  Otherwise, we
  // see if the name starts with '^', indicating a control edge. If we find
  // neither ':' nor '^' characters, the output index is implicitly 0, and the
  // whole name string forms the first part of the tensor name.
  const char* base = name.data();
  const char* p = base + name.size() - 1;
  unsigned int index = 0;
  unsigned int mul = 1;
  while (p > base && (*p >= '0' && *p <= '9')) {
    index += ((*p - '0') * mul);
    mul *= 10;
    p--;
  }
  TensorId id;
  if (p > base && *p == ':' && mul > 1) {
    id.first = absl::string_view(base, p - base);
    id.second = index;
  } else if (absl::StartsWith(name, "^")) {
    // Control edge
    id.first = absl::string_view(base + 1);
    id.second = Graph::kControlSlot;
  } else {
    id.first = name;
    id.second = 0;
  }
  return id;
}

bool IsTensorIdControl(const TensorId& tensor_id) {
  return tensor_id.index() == Graph::kControlSlot;
}

}  // namespace machina
