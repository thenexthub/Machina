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

#include "machina/xla/tsl/profiler/backends/cpu/threadpool_listener_state.h"

#include <atomic>

namespace tsl {
namespace profiler {
namespace threadpool_listener {
namespace {
static std::atomic<int> enabled = {0};
}

bool IsEnabled() { return enabled.load(std::memory_order_acquire); }

void Activate() { enabled.store(1, std::memory_order_release); }

void Deactivate() { enabled.store(0, std::memory_order_release); }

}  // namespace threadpool_listener
}  // namespace profiler
}  // namespace tsl
