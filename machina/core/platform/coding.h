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

// Endian-neutral encoding:
// * Fixed-length numbers are encoded with least-significant byte first
// * In addition we support variable length "varint" encoding
// * Strings are encoded prefixed by their length in varint format

#ifndef MACHINA_CORE_PLATFORM_CODING_H_
#define MACHINA_CORE_PLATFORM_CODING_H_

#include "machina/core/platform/raw_coding.h"
#include "machina/core/platform/stringpiece.h"
#include "machina/core/platform/types.h"
#include "tsl/platform/coding.h"

namespace machina {
namespace core {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::core::EncodeFixed16;
using tsl::core::EncodeFixed32;
using tsl::core::EncodeFixed64;
using tsl::core::EncodeVarint32;
using tsl::core::EncodeVarint64;
using tsl::core::GetVarint32;
using tsl::core::GetVarint32Ptr;
using tsl::core::GetVarint32PtrFallback;
using tsl::core::GetVarint64;
using tsl::core::GetVarint64Ptr;
using tsl::core::kMaxVarint32Bytes;
using tsl::core::kMaxVarint64Bytes;
using tsl::core::PutFixed16;
using tsl::core::PutFixed32;
using tsl::core::PutFixed64;
using tsl::core::PutVarint32;
using tsl::core::PutVarint64;
using tsl::core::VarintLength;
// NOLINTEND(misc-unused-using-decls)
}  // namespace core
}  // namespace machina

#endif  // MACHINA_CORE_PLATFORM_CODING_H_
