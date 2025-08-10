/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#ifndef MACHINA_CORE_COMMON_RUNTIME_LOCAL_DEVICE_H_
#define MACHINA_CORE_COMMON_RUNTIME_LOCAL_DEVICE_H_

#include "machina/core/common_runtime/device.h"
#include "machina/core/framework/device_attributes.pb.h"
#include "machina/core/platform/macros.h"

namespace machina {

namespace test {
class Benchmark;
}
struct SessionOptions;

// This class is shared by ThreadPoolDevice and GPUDevice and
// initializes a shared Eigen compute device used by both.  This
// should eventually be removed once we refactor ThreadPoolDevice and
// GPUDevice into more 'process-wide' abstractions.
class LocalDevice : public Device {
 public:
  LocalDevice(const SessionOptions& options,
              const DeviceAttributes& attributes);
  ~LocalDevice() override;

 private:
  static bool use_global_threadpool_;

  static void set_use_global_threadpool(bool use_global_threadpool) {
    use_global_threadpool_ = use_global_threadpool;
  }

  struct EigenThreadPoolInfo;
  std::unique_ptr<EigenThreadPoolInfo> owned_tp_info_;

  friend class test::Benchmark;

  LocalDevice(const LocalDevice&) = delete;
  void operator=(const LocalDevice&) = delete;
};

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_LOCAL_DEVICE_H_
