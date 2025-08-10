/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#ifndef MACHINA_CORE_KERNELS_RANDOM_INDEX_SHUFFLE_H_
#define MACHINA_CORE_KERNELS_RANDOM_INDEX_SHUFFLE_H_

#include <array>

#include "machina/core/platform/types.h"

namespace machina {
namespace random {

// Returns the position of `index` in a permutation of [0, ..., max_index].
//
// Index must be number in [0, ..., max_index].
// Key is the random key for the permutation.
// The returned index will also be in [0, ..., max_index]. For a fixed `key`
// and `max_index` the all possible `index` values and the returned values
// form a bijection.
// Rounds must be a positive even integer >= 4. Larger values increase improve
// 'randomness' of permutations for small `max_index` values. The time to
// compute the result scales linear with the number of rounds. We recommend 8
// rounds for a good treat off.
//
// For more details on the algorithm see the top of the cc file.
uint64_t index_shuffle(const uint64_t index, const std::array<uint32_t, 3>& key,
                       const uint64_t max_index, const int32_t rounds);

}  // namespace random
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_RANDOM_INDEX_SHUFFLE_H_
