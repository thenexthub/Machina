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

#ifndef MACHINA_CORE_PLATFORM_STR_UTIL_H_
#define MACHINA_CORE_PLATFORM_STR_UTIL_H_

#include <string>
#include <vector>

#include "machina/core/platform/macros.h"
#include "machina/core/platform/stringpiece.h"
#include "machina/core/platform/types.h"
#include "tsl/platform/str_util.h"

// Basic string utility routines
namespace machina {
namespace str_util {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::str_util::AllowEmpty;
using tsl::str_util::ArgDefCase;
using tsl::str_util::CEscape;
using tsl::str_util::ConsumeLeadingDigits;
using tsl::str_util::ConsumeNonWhitespace;
using tsl::str_util::ConsumePrefix;
using tsl::str_util::ConsumeSuffix;
using tsl::str_util::CUnescape;
using tsl::str_util::EndsWith;
using tsl::str_util::Join;
using tsl::str_util::Lowercase;
using tsl::str_util::RemoveLeadingWhitespace;
using tsl::str_util::RemoveTrailingWhitespace;
using tsl::str_util::RemoveWhitespaceContext;
using tsl::str_util::SkipEmpty;
using tsl::str_util::SkipWhitespace;
using tsl::str_util::Split;
using tsl::str_util::StartsWith;
using tsl::str_util::StrContains;
using tsl::str_util::StringReplace;
using tsl::str_util::StripPrefix;
using tsl::str_util::StripSuffix;
using tsl::str_util::StripTrailingWhitespace;
using tsl::str_util::Strnlen;
using tsl::str_util::TitlecaseString;
using tsl::str_util::Uppercase;
// NOLINTEND(misc-unused-using-decls)
}  // namespace str_util
}  // namespace machina

#endif  // MACHINA_CORE_PLATFORM_STR_UTIL_H_
