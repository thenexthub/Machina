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

#include "machina/core/graph/edgeset.h"

namespace machina {

std::pair<EdgeSet::const_iterator, bool> EdgeSet::insert(value_type value) {
  RegisterMutation();
  const_iterator ci;
  ci.Init(this);
  auto s = get_set();
  if (!s) {
    for (int i = 0; i < kInline; i++) {
      if (ptrs_[i] == value) {
        ci.array_iter_ = &ptrs_[i];
        return std::make_pair(ci, false);
      }
    }
    for (int i = 0; i < kInline; i++) {
      if (ptrs_[i] == nullptr) {
        ptrs_[i] = value;
        ci.array_iter_ = &ptrs_[i];
        return std::make_pair(ci, true);
      }
    }
    // array is full. convert to set.
    s = new gtl::FlatSet<const Edge*>;
    s->insert(reinterpret_cast<const Edge**>(std::begin(ptrs_)),
              reinterpret_cast<const Edge**>(std::end(ptrs_)));
    ptrs_[0] = this;
    ptrs_[1] = s;
    // fall through.
  }
  auto p = s->insert(value);
  ci.tree_iter_ = p.first;
  return std::make_pair(ci, p.second);
}

EdgeSet::size_type EdgeSet::erase(key_type key) {
  RegisterMutation();
  auto s = get_set();
  if (!s) {
    for (int i = 0; i < kInline; i++) {
      if (ptrs_[i] == key) {
        size_t n = size();
        ptrs_[i] = ptrs_[n - 1];
        ptrs_[n - 1] = nullptr;
        return 1;
      }
    }
    return 0;
  } else {
    return s->erase(key);
  }
}

}  // namespace machina
