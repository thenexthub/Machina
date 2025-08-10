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

#ifndef MACHINA_CORE_COMMON_RUNTIME_BFC_ALLOCATOR_H_
#define MACHINA_CORE_COMMON_RUNTIME_BFC_ALLOCATOR_H_

#include <array>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "machina/xla/tsl/framework/bfc_allocator.h"
#include "machina/core/common_runtime/allocator_retry.h"
#include "machina/core/common_runtime/shared_counter.h"
#include "machina/core/framework/allocator.h"
#include "machina/core/lib/strings/numbers.h"
#include "machina/core/lib/strings/strcat.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/platform/thread_annotations.h"
#include "machina/core/platform/types.h"

namespace machina {

class MemoryDump;         // NOLINT
using tsl::BFCAllocator;  // NOLINT

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_BFC_ALLOCATOR_H_
