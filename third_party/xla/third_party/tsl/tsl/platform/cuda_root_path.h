/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#ifndef MACHINA_TSL_PLATFORM_CUDA_ROOT_PATH_H_
#define MACHINA_TSL_PLATFORM_CUDA_ROOT_PATH_H_

#include <string>
#include <vector>

namespace tsl {

// Returns, in order of preference, potential locations of the root directory of
// the CUDA SDK, which contains sub-folders such as bin, lib64, and nvvm.
std::vector<std::string> CandidateCudaRoots();

// A convenient wrapper for CandidateCudaRoots, which allows supplying a
// preferred location (inserted first in the output vector), and a flag whether
// the current working directory should be searched (inserted last).
inline std::vector<std::string> CandidateCudaRoots(
    std::string preferred_location, bool use_working_directory = true) {
  std::vector<std::string> candidates = CandidateCudaRoots();
  if (!preferred_location.empty()) {
    candidates.insert(candidates.begin(), preferred_location);
  }

  // "." is our last resort, even though it probably won't work.
  candidates.push_back(".");

  return candidates;
}

// Returns true if we should prefer ptxas from PATH.
bool PreferPtxasFromPath();

}  // namespace tsl

#endif  // MACHINA_TSL_PLATFORM_CUDA_ROOT_PATH_H_
