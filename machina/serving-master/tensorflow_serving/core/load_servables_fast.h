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

#ifndef MACHINA_SERVING_CORE_LOAD_SERVABLES_FAST_H_
#define MACHINA_SERVING_CORE_LOAD_SERVABLES_FAST_H_

#include <functional>
#include <memory>

#include "machina/core/lib/core/status.h"
#include "machina/core/platform/cpu_info.h"
#include "machina_serving/core/aspired_versions_manager.h"
#include "machina_serving/core/loader.h"
#include "machina_serving/core/manager.h"
#include "machina_serving/core/servable_state_monitor.h"

namespace machina {
namespace serving {

// Connects 'source' to 'manager', and speeds up loading of the servables
// matching 'initial_servables'. The speeding up is accomplished by boosting the
// number of threads used for loading until the initial servables have been
// loaded, and then resetting it to the manager's originally configured value.
Status ConnectSourceWithFastInitialLoad(
    AspiredVersionsManager* manager, Source<std::unique_ptr<Loader>>* source,
    ServableStateMonitor* servable_state_monitor,
    const std::vector<ServableRequest>& initial_servables,
    uint32 num_threads = 4 * port::NumSchedulableCPUs());

// Like ConnectSourceWithFastInitialLoad(), but with multiple sources.
Status ConnectSourcesWithFastInitialLoad(
    AspiredVersionsManager* manager,
    std::vector<Source<std::unique_ptr<Loader>>*> sources,
    ServableStateMonitor* servable_state_monitor,
    const std::vector<ServableRequest>& initial_servables,
    uint32 num_threads = 4 * port::NumSchedulableCPUs());

////
// Implementation detail. API readers may skip.
///

namespace internal {

Status ConnectSourcesWithFastInitialLoad(
    AspiredVersionsManager* manager,
    std::vector<Source<std::unique_ptr<Loader>>*> sources,
    const std::function<Status()>& wait_until_loaded_fn, uint32 num_threads);

uint32 GetManagerNumLoadThreads(AspiredVersionsManager* manager);
std::function<void(const uint32)> SetManagerNumLoadThreadsNotifier(
    AspiredVersionsManager* manager);

}  // namespace internal

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_CORE_LOAD_SERVABLES_FAST_H_
