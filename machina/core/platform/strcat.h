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

#ifndef MACHINA_CORE_PLATFORM_STRCAT_H_
#define MACHINA_CORE_PLATFORM_STRCAT_H_

#include "machina/core/platform/macros.h"
#include "machina/core/platform/numbers.h"
#include "machina/core/platform/stringpiece.h"
#include "machina/core/platform/types.h"
#include "tsl/platform/strcat.h"

namespace machina {
namespace strings {

// NOLINTBEGIN(misc-unused-using-decls)
using tsl::strings::AlphaNum;
using tsl::strings::Hex;
using tsl::strings::PadSpec;
using tsl::strings::StrAppend;
using tsl::strings::StrCat;
// NOLINTEND(misc-unused-using-decls)

}  // namespace strings
}  // namespace machina

#endif  // MACHINA_CORE_PLATFORM_STRCAT_H_
