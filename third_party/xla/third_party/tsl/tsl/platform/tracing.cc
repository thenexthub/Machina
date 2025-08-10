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

#include "tsl/platform/tracing.h"

#include <array>
#include <atomic>

#include "tsl/platform/hash.h"

namespace tsl {
namespace tracing {
namespace {
std::atomic<uint64> unique_arg{1};
}  // namespace

const char* GetEventCategoryName(EventCategory category) {
  switch (category) {
    case EventCategory::kScheduleClosure:
      return "ScheduleClosure";
    case EventCategory::kRunClosure:
      return "RunClosure";
    case EventCategory::kCompute:
      return "Compute";
    default:
      return "Unknown";
  }
}

std::array<const EventCollector*, GetNumEventCategories()>
    EventCollector::instances_;

void SetEventCollector(EventCategory category,
                       const EventCollector* collector) {
  EventCollector::instances_[static_cast<unsigned>(category)] = collector;
}

uint64 GetUniqueArg() {
  return unique_arg.fetch_add(1, std::memory_order_relaxed);
}

uint64 GetArgForName(absl::string_view name) {
  return Hash64(name.data(), name.size());
}

}  // namespace tracing
}  // namespace tsl
