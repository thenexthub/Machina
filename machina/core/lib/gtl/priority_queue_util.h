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

#ifndef MACHINA_CORE_LIB_GTL_PRIORITY_QUEUE_UTIL_H_
#define MACHINA_CORE_LIB_GTL_PRIORITY_QUEUE_UTIL_H_

#include <algorithm>
#include <queue>
#include <utility>

namespace machina {
namespace gtl {

// Removes the top element from a std::priority_queue and returns it.
// Supports movable types.
template <typename T, typename Container, typename Comparator>
T ConsumeTop(std::priority_queue<T, Container, Comparator>* q) {
  // std::priority_queue is required to implement pop() as if it
  // called:
  //   std::pop_heap()
  //   c.pop_back()
  // unfortunately, it does not provide access to the removed element.
  // If the element is move only (such as a unique_ptr), there is no way to
  // reclaim it in the standard API.  std::priority_queue does, however, expose
  // the underlying container as a protected member, so we use that access
  // to extract the desired element between those two calls.
  using Q = std::priority_queue<T, Container, Comparator>;
  struct Expose : Q {
    using Q::c;
    using Q::comp;
  };
  auto& c = q->*&Expose::c;
  auto& comp = q->*&Expose::comp;
  std::pop_heap(c.begin(), c.end(), comp);
  auto r = std::move(c.back());
  c.pop_back();
  return r;
}

}  // namespace gtl
}  // namespace machina

#endif  // MACHINA_CORE_LIB_GTL_PRIORITY_QUEUE_UTIL_H_
