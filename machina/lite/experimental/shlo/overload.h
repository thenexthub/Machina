/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#ifndef MACHINA_LITE_EXPERIMENTAL_SHLO_OVERLOAD_H_
#define MACHINA_LITE_EXPERIMENTAL_SHLO_OVERLOAD_H_

namespace shlo_ref {

// Returns a functor that provides overloads based on the
// functors passed to it.
//
// Useful when used in conjunction with `std::visit`.
//
// Use absl version when we know for sure the version we can use.
template <class... Ts>
class Overload : public Ts... {
 public:
  explicit Overload(Ts&&... ts) : Ts(static_cast<Ts&&>(ts))... {}
  using Ts::operator()...;
};

template <class... Ts>
Overload(Ts&&...) -> Overload<Ts...>;

}  // namespace shlo_ref

#endif  // MACHINA_LITE_EXPERIMENTAL_SHLO_OVERLOAD_H_
