/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#ifndef MACHINA_LITE_DELEGATES_GPU_COMMON_TRANSFORMATIONS_MATCHING_H_
#define MACHINA_LITE_DELEGATES_GPU_COMMON_TRANSFORMATIONS_MATCHING_H_

// A file provides predicates to match subgraphs.

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

namespace tflite {
namespace gpu {

// Returns true if a container of nodes contains nodes that all match given
// operation_types.
template <typename T>
bool MatchesByOperationType(const T& nodes,
                            const std::vector<std::string>& types) {
  if (nodes.size() != types.size()) return false;
  return std::mismatch(nodes.begin(), nodes.end(), types.begin(),
                       [&](typename T::value_type a, const std::string& b) {
                         return a->operation.type == b;
                       })
             .first == nodes.end();
}

}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_COMMON_TRANSFORMATIONS_MATCHING_H_
