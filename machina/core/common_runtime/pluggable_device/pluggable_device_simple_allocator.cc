/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#include "machina/core/common_runtime/pluggable_device/pluggable_device_simple_allocator.h"

#include <cstddef>
#include <optional>

#include "machina/xla/tsl/framework/allocator.h"
#include "machina/core/framework/allocator.h"

namespace machina {

PluggableDeviceSimpleAllocator::PluggableDeviceSimpleAllocator(
    tsl::SubAllocator* sub_allocator)
    : sub_allocator_(sub_allocator) {}

void* PluggableDeviceSimpleAllocator::AllocateRaw(size_t alignment,
                                                  size_t num_bytes) {
  size_t bytes_received;
  return sub_allocator_->Alloc(alignment, num_bytes, &bytes_received);
}

void PluggableDeviceSimpleAllocator::DeallocateRaw(void* ptr) {
  return sub_allocator_->Free(ptr, 0);
}

std::optional<AllocatorStats> PluggableDeviceSimpleAllocator::GetStats() {
  AllocatorStats stats_;
  stats_.num_allocs = 0;
  stats_.peak_bytes_in_use = 0;
  stats_.largest_alloc_size = 0;

  return stats_;
}

}  // namespace machina
