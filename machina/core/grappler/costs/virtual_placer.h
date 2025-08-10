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

#ifndef MACHINA_CORE_GRAPPLER_COSTS_VIRTUAL_PLACER_H_
#define MACHINA_CORE_GRAPPLER_COSTS_VIRTUAL_PLACER_H_

#include <unordered_map>

#include "machina/core/platform/types.h"
#include "machina/core/protobuf/device_properties.pb.h"

namespace machina {
class NodeDef;

namespace grappler {
class Cluster;

// The virtual placer emulates the behavior of the TF placer.
class VirtualPlacer {
 public:
  explicit VirtualPlacer(
      const std::unordered_map<string, DeviceProperties>& devices);

  const DeviceProperties& get_device(const NodeDef& node) const;

  // Returns device name from cluster, which best matches the node.device()
  // specification. Returns default device if no match was found or the
  // node.device() could not be parsed.
  string get_canonical_device_name(const NodeDef& node) const;

 private:
  // Converts given device name to Lowercase Fully-Qualified Name (LFQN) string.
  // This helps us disambiguate device names internally and simplify matching.
  // If device_name couldn't be parsed successfully, returns empty string.
  string to_lfqn_or_empty(const string& device_name) const;

  // Map based on the cluster info: cluster device name -> device properties.
  std::unordered_map<string, DeviceProperties> devices_;

  // Maps LFQN to original device name as it was declared in cluster.
  std::unordered_map<string, string> lfqn_map_;

  string default_device_name_;
  string default_job_name_lowercase_;
};

}  // namespace grappler
}  // end namespace machina

#endif  // MACHINA_CORE_GRAPPLER_COSTS_VIRTUAL_PLACER_H_
