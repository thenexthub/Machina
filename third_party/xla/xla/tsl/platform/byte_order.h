/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#ifndef MACHINA_XLATSL_PLATFORM_BYTE_ORDER_H_
#define MACHINA_XLATSL_PLATFORM_BYTE_ORDER_H_

// Byte order defines provided by gcc. MSVC doesn't define those so
// we define them here.
// We assume that all windows platform out there are little endian.
#if defined(_MSC_VER) && !defined(__clang__)
#define __ORDER_LITTLE_ENDIAN__ 0x4d2
#define __ORDER_BIG_ENDIAN__ 0x10e1
#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
#endif

namespace tsl {
namespace port {

// TODO(jeff,sanjay): Make portable
constexpr bool kLittleEndian = __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;

}  // namespace port
}  // namespace tsl

#endif  // MACHINA_XLATSL_PLATFORM_BYTE_ORDER_H_
