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

#ifndef MACHINA_TSL_PLATFORM_LOAD_LIBRARY_H_
#define MACHINA_TSL_PLATFORM_LOAD_LIBRARY_H_

#include <string>

#include "absl/status/status.h"

namespace tsl {

namespace internal {

absl::Status LoadDynamicLibrary(const char* library_filename, void** handle);
absl::Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                                  void** symbol);
std::string FormatLibraryFileName(const std::string& name,
                                  const std::string& version);

}  // namespace internal

}  // namespace tsl

#endif  // MACHINA_TSL_PLATFORM_LOAD_LIBRARY_H_
