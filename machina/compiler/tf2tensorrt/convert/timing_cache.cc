/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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
#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "machina/compiler/tf2tensorrt/convert/timing_cache.h"

#include <unordered_map>

#include "machina/compiler/tf2tensorrt/common/utils.h"
#include "machina/core/platform/errors.h"
#include "third_party/tensorrt/NvInfer.h"

namespace machina {
namespace tensorrt {
namespace convert {

StatusOr<TimingCacheRegistry::TimingCachePtr> TimingCacheRegistry::LookUp(
    const string& name, nvinfer1::IBuilderConfig* builder_config) {
#if IS_TRT_VERSION_GE(8, 0, 0, 0)
  TRT_ENSURE(builder_config != nullptr);
  mutex_lock scoped_lock(mu_);
  if (map_.find(name) != map_.end()) {
    const std::vector<uint8_t>& data = map_[name];
    return std::unique_ptr<nvinfer1::ITimingCache>(
        builder_config->createTimingCache(data.data(), data.size()));
  }

  // If no such timing cache exists, create a new timing cache.
  return std::unique_ptr<nvinfer1::ITimingCache>(
      builder_config->createTimingCache(nullptr, 0));
#endif  // IS_TRT_VERSION_GE(8, 0, 0, 0)
  return errors::Unimplemented(
      "serializable timing cache does not exist in TensorRT versions < 8.0");
}

void TimingCacheRegistry::Upsert(const string& name, TimingCache* cache) {
#if IS_TRT_VERSION_GE(8, 0, 0, 0)
  nvinfer1::IHostMemory* memory = cache->serialize();
  if (memory == nullptr) {
    return;
  }

  if (map_.find(name) == map_.end()) {
    // If the timing cache with the given name does not exist, emplace the
    // serialized buffer.
    std::vector<uint8_t> mem(memory->size());
    std::copy_n(static_cast<uint8_t*>(memory->data()), memory->size(),
                mem.begin());
    {
      mutex_lock scoped_lock(mu_);
      map_.emplace(name, std::move(mem));
    }
  } else {
    // If the timing cache does exist, use the existing buffer.
    mutex_lock scoped_lock(mu_);
    std::vector<uint8_t>& mem = map_[name];
    mem.resize(memory->size());
    std::copy_n(static_cast<uint8_t*>(memory->data()), memory->size(),
                mem.begin());
  }
  memory->destroy();
#endif  // IS_TRT_VERSION_GE(8, 0, 0, 0)
}

TimingCacheRegistry* GetTimingCacheRegistry() {
  static TimingCacheRegistry* registry = new TimingCacheRegistry();
  return registry;
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace machina

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
