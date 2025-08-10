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

#ifndef MACHINA_TSL_PLATFORM_SNAPPY_H_
#define MACHINA_TSL_PLATFORM_SNAPPY_H_

#include "machina/xla/tsl/platform/types.h"

#if !defined(PLATFORM_WINDOWS)
#include <sys/uio.h>
namespace tsl {
using ::iovec;  // NOLINT(misc-unused-using-decls)
}  // namespace tsl
#else
namespace tsl {
struct iovec {
  void* iov_base;
  size_t iov_len;
};
}  // namespace tsl
#endif

namespace tsl {
namespace port {

// Snappy compression/decompression support
bool Snappy_Compress(const char* input, size_t length, string* output);

bool Snappy_CompressFromIOVec(const struct iovec* iov,
                              size_t uncompressed_length, string* output);

bool Snappy_GetUncompressedLength(const char* input, size_t length,
                                  size_t* result);
bool Snappy_Uncompress(const char* input, size_t length, char* output);

bool Snappy_UncompressToIOVec(const char* compressed, size_t compressed_length,
                              const struct iovec* iov, size_t iov_cnt);

}  // namespace port
}  // namespace tsl

#endif  // MACHINA_TSL_PLATFORM_SNAPPY_H_
