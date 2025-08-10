/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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
#include "machina/tools/proto_splitter/cc/max_size.h"

#include "absl/synchronization/mutex.h"

namespace machina {
namespace tools::proto_splitter {
ABSL_CONST_INIT absl::Mutex global_mutex(absl::kConstInit);

// The default max size is set to a bit less than 2GB, since the proto splitter
// isn't extremely precise.
uint64_t ProtoMaxSize = ((uint64_t)1 << 31) - 500;

uint64_t GetMaxSize() { return ProtoMaxSize; }

void DebugSetMaxSize(uint64_t size) { ProtoMaxSize = size; }

}  // namespace tools::proto_splitter
}  // namespace machina
