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

// Printf variants that place their output in a C++ string.
//
// Usage:
//      string result = strings::Printf("%d %s\n", 10, "hello");
//      strings::Appendf(&result, "%d %s\n", 20, "there");

#ifndef MACHINA_TSL_PLATFORM_STRINGPRINTF_H_
#define MACHINA_TSL_PLATFORM_STRINGPRINTF_H_

#include <stdarg.h>

#include <string>

#include "machina/xla/tsl/platform/macros.h"
#include "machina/xla/tsl/platform/types.h"

namespace tsl {
namespace strings {

// Return a C++ string
std::string Printf(const char* format, ...)
    // Tell the compiler to do printf format string checking.
    TF_PRINTF_ATTRIBUTE(1, 2);

// Append result to a supplied string
void Appendf(std::string* dst, const char* format, ...)
    // Tell the compiler to do printf format string checking.
    TF_PRINTF_ATTRIBUTE(2, 3);

// Lower-level routine that takes a va_list and appends to a specified
// string.  All other routines are just convenience wrappers around it.
void Appendv(std::string* dst, const char* format, va_list ap);

}  // namespace strings
}  // namespace tsl

#endif  // MACHINA_TSL_PLATFORM_STRINGPRINTF_H_
