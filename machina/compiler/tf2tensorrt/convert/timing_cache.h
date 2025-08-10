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
#ifndef MACHINA_COMPILER_TF2TENSORRT_CONVERT_TIMING_CACHE_H_
#define MACHINA_COMPILER_TF2TENSORRT_CONVERT_TIMING_CACHE_H_
#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include <unordered_map>

#include "machina/compiler/tf2tensorrt/common/utils.h"
#include "machina/core/framework/registration/registration.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/platform/statusor.h"
#include "third_party/tensorrt/NvInfer.h"

namespace machina {
namespace tensorrt {
namespace convert {

// A registry for holding serialized TensorRT autotuner timing caches.
// For TensorRT versions < 8.0, the timing cache is not serializable, so these
// operations become no-ops.
class TimingCacheRegistry {
 public:
  TimingCacheRegistry() = default;
  ~TimingCacheRegistry() = default;

#if IS_TRT_VERSION_GE(8, 0, 0, 0)
  using TimingCache = nvinfer1::ITimingCache;
  using TimingCachePtr = std::unique_ptr<TimingCache>;
#else
  struct TimingCache {};
  using TimingCachePtr = std::unique_ptr<TimingCache>;
#endif

  // Insert or update a registry into the map using the given name. The cache
  // will be serialized before being placed into the map.
  void Upsert(const string& name, TimingCache* cache);

  // Find a timing cache using the given name. The provided BuilderConfig is
  // used to deserialize the cache. If no timing cache is found, a new timing
  // cache is returned.
  StatusOr<TimingCachePtr> LookUp(const string& name,
                                  nvinfer1::IBuilderConfig* builder_config);

 private:
  using SerializedTimingCache = std::vector<uint8_t>;

  mutex mu_;
  std::unordered_map<std::string, SerializedTimingCache> map_;
};

TimingCacheRegistry* GetTimingCacheRegistry();

}  // namespace convert
}  // namespace tensorrt
}  // namespace machina

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // MACHINA_COMPILER_TF2TENSORRT_CONVERT_TIMING_CACHE_H_
