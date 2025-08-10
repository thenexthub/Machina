/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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
#include <string>

#include "tsl/profiler/lib/nvtx_utils.h"

namespace tsl::profiler {
ProfilerDomainHandle DefaultProfilerDomain() { return {}; }
void NameCurrentThread(const std::string&) {}
void NameDevice(int, const std::string&) {}
void NameStream(StreamHandle, const std::string&) {}
void RangePop(ProfilerDomainHandle) {}
void RangePush(ProfilerDomainHandle, const char*) {}
namespace detail {
void RangePush(ProfilerDomainHandle, StringHandle, uint64_t, const void*,
               size_t) {}
}  // namespace detail
uint64_t RegisterSchema(ProfilerDomainHandle, const void*) { return 0; }
StringHandle RegisterString(ProfilerDomainHandle, const std::string&) {
  return {};
}
void MarkMemoryInitialized(void const* address, size_t size,
                           StreamHandle stream) {}
}  // namespace tsl::profiler
