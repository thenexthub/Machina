/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
// This file is used to provide equivalents of internal absl::FormatF
// and absl::StrAppendFormat. Unfortunately, type safety is not as good as a
// a full C++ example.
// TODO(aselle): When absl adds support for StrFormat, use that instead.
#ifndef MACHINA_LITE_TOCO_FORMAT_PORT_H_
#define MACHINA_LITE_TOCO_FORMAT_PORT_H_

#include <string>

#include "machina/core/lib/strings/stringprintf.h"
#include "machina/lite/toco/toco_types.h"

namespace toco {
namespace port {

/// Identity (default case)
template <class T>
T IdentityOrConvertStringToRaw(T foo) {
  return foo;
}

// Overloaded case where we return std::string.
inline const char* IdentityOrConvertStringToRaw(const std::string& foo) {
  return foo.c_str();
}

// Delegate to TensorFlow Appendf function until absl has an equivalent.
template <typename... Args>
inline void AppendFHelper(std::string* destination, const char* fmt,
                          Args&&... args) {
  machina::strings::Appendf(destination, fmt, args...);
}

// Specialization for no argument format string (avoid security bug).
inline void AppendFHelper(std::string* destination, const char* fmt) {
  machina::strings::Appendf(destination, "%s", fmt);
}

// Append formatted string (with format fmt and args args) to the string
// pointed to by destination. fmt follows C printf semantics.
// One departure is that %s can be driven by a std::string or string.
template <typename... Args>
inline void AppendF(std::string* destination, const char* fmt, Args&&... args) {
  AppendFHelper(destination, fmt, IdentityOrConvertStringToRaw(args)...);
}

// Return formatted string (with format fmt and args args). fmt follows C printf
// semantics. One departure is that %s can be driven by a std::string or string.
template <typename... Args>
inline std::string StringF(const char* fmt, Args&&... args) {
  std::string result;
  AppendFHelper(&result, fmt, IdentityOrConvertStringToRaw(args)...);
  return result;
}

}  // namespace port
}  // namespace toco

#endif  // MACHINA_LITE_TOCO_FORMAT_PORT_H_
