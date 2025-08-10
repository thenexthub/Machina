/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

#ifndef MACHINA_CORE_GRAPPLER_DEVICES_H_
#define MACHINA_CORE_GRAPPLER_DEVICES_H_

#include <cstdint>
#include <functional>
#include <utility>

#include "machina/core/lib/core/status.h"
#include "machina/core/lib/core/threadpool.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace grappler {

// Get the number of available GPUs whose number of multiprocessors is no less
// than 8 and whose CUDA compute capability is no less than
// min_cuda_compute_capability.
int GetNumAvailableGPUs(
    const std::pair<int, int>& min_cuda_compute_capability = {0, 0});

// Maximum amount of gpu memory available per gpu. gpu_id must be in the range
// [0, num_available_gpu)
int64_t AvailableGPUMemory(int gpu_id);

// Get the number of logical CPU cores (aka hyperthreads) available.
int GetNumAvailableLogicalCPUCores();

}  // end namespace grappler
}  // end namespace machina

#endif  // MACHINA_CORE_GRAPPLER_DEVICES_H_
