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

#ifndef MACHINA_SERVING_CORE_RESOURCE_PRESERVING_POLICY_H_
#define MACHINA_SERVING_CORE_RESOURCE_PRESERVING_POLICY_H_

#include <vector>

#include "absl/types/optional.h"
#include "machina_serving/core/aspired_version_policy.h"
#include "machina_serving/core/loader_harness.h"

namespace machina {
namespace serving {

// ServablePolicy that eagerly unloads any no-longer-aspired versions of a
// servable stream and only after done unloading, loads newly aspired versions
// in the order of descending version number.
//
// This policy minimizes resource consumption with the trade-off of temporary
// servable unavailability while all old versions unload followed by the new
// versions loading.
//
// Servables with a single version consuming the majority of their host's
// resources must use this policy to prevent deadlock. Other typical use-cases
// will be for multi-servable environments where clients can tolerate brief
// interruptions to a single servable's availability on a replica.
//
// NB: This policy does not in any way solve cross-replica availability.
class ResourcePreservingPolicy final : public AspiredVersionPolicy {
 public:
  absl::optional<ServableAction> GetNextAction(
      const std::vector<AspiredServableStateSnapshot>& all_versions)
      const override;
};

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_CORE_RESOURCE_PRESERVING_POLICY_H_
