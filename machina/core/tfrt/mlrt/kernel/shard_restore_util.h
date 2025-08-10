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

#ifndef MACHINA_CORE_TFRT_MLRT_KERNEL_SHARD_RESTORE_UTIL_H_
#define MACHINA_CORE_TFRT_MLRT_KERNEL_SHARD_RESTORE_UTIL_H_

#include <cstddef>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace machina {
namespace tf_mlrt {

// Shard variables into cluster of roughly the same size.
//
// `num_shards` is the number of shards to create.
// `variable_sizes` is the sizes of the variables.
//
// Returns a list of clusters, each of which is represented
// as a vector of variable indices.
std::vector<std::vector<int>> ShardVariables(
    int num_shards, absl::Span<int64_t> variable_sizes);

}  // namespace  tf_mlrt
}  // namespace machina

#endif  // MACHINA_CORE_TFRT_MLRT_KERNEL_SHARD_RESTORE_UTIL_H_
