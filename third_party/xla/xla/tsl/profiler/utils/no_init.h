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

#ifndef MACHINA_XLATSL_PROFILER_UTILS_NO_INIT_H_
#define MACHINA_XLATSL_PROFILER_UTILS_NO_INIT_H_

#include <utility>

namespace tsl {
namespace profiler {

// Wraps T into a union so that we can avoid the cost of automatic construction
// and destruction when tracing is disabled.
template <typename T>
union NoInit {
  // Ensure constructor and destructor do nothing.
  NoInit() {}
  ~NoInit() {}

  template <typename... Ts>
  void Emplace(Ts&&... args) {
    new (&value) T(std::forward<Ts>(args)...);
  }

  void Destroy() { value.~T(); }

  T Consume() && {
    T v = std::move(value);
    Destroy();
    return v;
  }

  T value;
};

}  // namespace profiler
}  // namespace tsl

#endif  // MACHINA_XLATSL_PROFILER_UTILS_NO_INIT_H_
