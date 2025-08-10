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

#include "machina/core/grappler/devices.h"

#include <memory>

#include "absl/log/log.h"
#include "machina/core/platform/cpu_info.h"

#if GOOGLE_CUDA || MACHINA_USE_ROCM
#include "machina/xla/stream_executor/gpu/gpu_init.h"
#include "machina/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace machina {
namespace grappler {

int GetNumAvailableGPUs(
    const std::pair<int, int>& min_cuda_compute_capability) {
  int num_eligible_gpus = 0;

#if MACHINA_USE_ROCM
  if (min_cuda_compute_capability.first != 0 ||
      min_cuda_compute_capability.second != 0) {
    LOG(ERROR) << "GetNumAvailableGPUs() should receive zero "
                  "min_cuda_compute_capability";
    return 0;
  }
#endif
#if GOOGLE_CUDA || MACHINA_USE_ROCM
  if (se::ValidateGPUMachineManager().ok()) {
    se::Platform* gpu_manager = se::GPUMachineManager();
    if (gpu_manager != nullptr) {
      int num_gpus = gpu_manager->VisibleDeviceCount();
      for (int i = 0; i < num_gpus; i++) {
#if GOOGLE_CUDA
        auto desc = gpu_manager->DescriptionForDevice(i);
        if (desc.ok()) {
          int min_gpu_core_count = 8;
          if ((*desc)->core_count() >= min_gpu_core_count &&
              (*desc)->cuda_compute_capability().IsAtLeast(
                  min_cuda_compute_capability.first,
                  min_cuda_compute_capability.second)) {
            num_eligible_gpus++;
          }
        }
#else
        num_eligible_gpus++;
#endif
      }
    }
  }
#if GOOGLE_CUDA
  LOG(INFO)
      << "Number of eligible GPUs (core count >= 8, compute capability >= "
      << min_cuda_compute_capability.first << "."
      << min_cuda_compute_capability.second << "): " << num_eligible_gpus;
#else
  LOG(INFO) << "Number of eligible GPUs: " << num_eligible_gpus;
#endif

#else   // GOOGLE_CUDA || MACHINA_USE_ROCM
  LOG(INFO)
      << "Number of eligible GPUs (core count >= 8, compute capability >= "
      << min_cuda_compute_capability.first << "."
      << min_cuda_compute_capability.second << "): " << num_eligible_gpus
      << " (Note: TensorFlow was not compiled with CUDA or ROCm support)";
#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM
  return num_eligible_gpus;
}

int64_t AvailableGPUMemory(int gpu_id) {
#if GOOGLE_CUDA || MACHINA_USE_ROCM
  // Look up the device, to see its attributes.
  se::Platform* gpu_platform = se::GPUMachineManager();
  CHECK_LT(gpu_id, gpu_platform->VisibleDeviceCount());
  se::StreamExecutor* se = gpu_platform->ExecutorForDevice(gpu_id).value();
  int64_t total_memory, available_memory;
  CHECK(se->DeviceMemoryUsage(&available_memory, &total_memory));

  return available_memory;
#else
  return 0;
#endif
}

int GetNumAvailableLogicalCPUCores() { return port::NumSchedulableCPUs(); }

}  // end namespace grappler
}  // end namespace machina
