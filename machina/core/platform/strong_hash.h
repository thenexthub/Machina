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

#ifndef MACHINA_CORE_PLATFORM_STRONG_HASH_H_
#define MACHINA_CORE_PLATFORM_STRONG_HASH_H_

#include "highwayhash/sip_hash.h"  // from @highwayhash
#include "highwayhash/state_helpers.h"  // from @highwayhash
#include "machina/core/platform/platform.h"
#include "machina/core/platform/types.h"

namespace machina {

// This is a strong keyed hash function interface for strings.
// The hash function is deterministic on the content of the string within the
// process. The key of the hash is an array of 2 uint64 elements.
// A strong hash makes it difficult, if not infeasible, to compute inputs that
// hash to the same bucket.
//
// Usage:
//   uint64 key[2] = {123, 456};
//   string input = "input string";
//   uint64 hash_value = StrongKeyedHash(key, input);
//
inline uint64 StrongKeyedHash(const machina::uint64 (&key)[2],
                              const string& s) {
  return highwayhash::StringHasher<highwayhash::SipHashState>()(
      {key[0], key[1]}, s);
}

}  // namespace machina

#endif  // MACHINA_CORE_PLATFORM_STRONG_HASH_H_
