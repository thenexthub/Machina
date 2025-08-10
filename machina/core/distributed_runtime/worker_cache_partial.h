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

#ifndef MACHINA_CORE_DISTRIBUTED_RUNTIME_WORKER_CACHE_PARTIAL_H_
#define MACHINA_CORE_DISTRIBUTED_RUNTIME_WORKER_CACHE_PARTIAL_H_

#include <string>
#include <unordered_map>

#include "machina/core/distributed_runtime/worker_cache.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/platform/thread_annotations.h"
#include "machina/core/platform/types.h"
#include "machina/core/protobuf/worker.pb.h"

namespace machina {

// Implements the part of the interface that caches and returns remote
// device status attributes.
class WorkerCachePartial : public WorkerCacheInterface {
 public:
  bool GetDeviceLocalityNonBlocking(const string& device,
                                    DeviceLocality* locality) override;

  void GetDeviceLocalityAsync(const string& device, DeviceLocality* locality,
                              StatusCallback) override;

  ~WorkerCachePartial() override {}

  // Clear all entries from the DeviceStatus cache.
  void FlushStatusCache();

 private:
  mutex mu_;

  // Initiate a GetStatusAsync to the remote task named by "task", and
  // update the cache with all the DeviceAttributes reported.
  absl::Status RefreshDeviceStatus(const string& device_name);

  typedef std::unordered_map<string, DeviceAttributes> StatusMap;
  StatusMap device_status_cache_ TF_GUARDED_BY(mu_);
};

}  // namespace machina
#endif  // MACHINA_CORE_DISTRIBUTED_RUNTIME_WORKER_CACHE_PARTIAL_H_
