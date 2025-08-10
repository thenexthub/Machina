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

// Methods related to the availability of servables, that are useful in writing
// tests. (Not intended for production use.)

#ifndef MACHINA_SERVING_CORE_TEST_UTIL_AVAILABILITY_TEST_UTIL_H_
#define MACHINA_SERVING_CORE_TEST_UTIL_AVAILABILITY_TEST_UTIL_H_

#include "machina_serving/core/servable_state_monitor.h"

namespace machina {
namespace serving {
namespace test_util {

// Waits until 'monitor' shows that the manager state of 'servable' is one of
// 'states'.
void WaitUntilServableManagerStateIsOneOf(
    const ServableStateMonitor& monitor, const ServableId& servable,
    const std::vector<ServableState::ManagerState>& states);

// Waits until 'monitor' shows that the manager state servable ids is
// kAvailable.
void WaitUntilVersionsAvailable(const ServableStateMonitor& monitor,
                                const string& servable_id_name,
                                absl::Span<const int64_t> servable_id_versions);

}  // namespace test_util
}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_CORE_TEST_UTIL_AVAILABILITY_TEST_UTIL_H_
