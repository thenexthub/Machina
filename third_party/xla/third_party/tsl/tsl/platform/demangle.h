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

#ifndef MACHINA_TSL_PLATFORM_DEMANGLE_H_
#define MACHINA_TSL_PLATFORM_DEMANGLE_H_

#include "machina/xla/tsl/platform/types.h"

namespace tsl {
namespace port {

// If the compiler supports, demangle a mangled symbol name and return
// the demangled name. Otherwise, returns 'mangled' as is.
string Demangle(const char* mangled);
inline string Demangle(const string mangled) {
  return Demangle(mangled.c_str());
}

}  // namespace port
}  // namespace tsl

#endif  // MACHINA_TSL_PLATFORM_DEMANGLE_H_
