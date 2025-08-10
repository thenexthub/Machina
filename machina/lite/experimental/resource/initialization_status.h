/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#ifndef MACHINA_LITE_EXPERIMENTAL_RESOURCE_INITIALIZATION_STATUS_H_
#define MACHINA_LITE_EXPERIMENTAL_RESOURCE_INITIALIZATION_STATUS_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>

#include "machina/lite/core/c/common.h"
#include "machina/lite/experimental/resource/resource_base.h"

namespace tflite {
namespace resource {

/// WARNING: Experimental interface, subject to change.
// An initialization status class. This class will record the completion status
// of the initialization procedure. For example, when the initialization
// subgraph should be invoked  once in a life cycle, this class instance will
// have the initialization status in order to make sure the followup invocations
// to invoke the initalization subgraph can be ignored safely.
class InitializationStatus : public ResourceBase {
 public:
  InitializationStatus() {}
  InitializationStatus(InitializationStatus&& other) noexcept {
    is_initialized_ = other.is_initialized_;
  }

  InitializationStatus(const InitializationStatus&) = delete;
  InitializationStatus& operator=(const InitializationStatus&) = delete;

  ~InitializationStatus() override {}

  // Mark initialization is done.
  void MarkInitializationIsDone();

  // Returns true if this initialization is done.
  bool IsInitialized() override;

  size_t GetMemoryUsage() override { return 0; }

 private:
  // True if the initialization process is done.
  bool is_initialized_ = false;
};

/// WARNING: Experimental interface, subject to change.
using InitializationStatusMap =
    std::unordered_map<std::int32_t, std::unique_ptr<InitializationStatus>>;

InitializationStatus* GetInitializationStatus(InitializationStatusMap* map,
                                              int subgraph_id);

}  // namespace resource
}  // namespace tflite

#endif  // MACHINA_LITE_EXPERIMENTAL_RESOURCE_INITIALIZATION_STATUS_H_
