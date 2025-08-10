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

#include "machina/lite/experimental/resource/initialization_status.h"

#include <memory>

namespace tflite {
namespace resource {

void InitializationStatus::MarkInitializationIsDone() {
  is_initialized_ = true;
}

bool InitializationStatus::IsInitialized() { return is_initialized_; }

InitializationStatus* GetInitializationStatus(InitializationStatusMap* map,
                                              int subgraph_id) {
  auto it = map->find(subgraph_id);
  if (it != map->end()) {
    return static_cast<InitializationStatus*>(it->second.get());
  }

  InitializationStatus* status = new InitializationStatus();
  map->emplace(subgraph_id, std::unique_ptr<InitializationStatus>(status));
  return status;
}

}  // namespace resource
}  // namespace tflite
