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

#ifndef MACHINA_CORE_DEBUG_DEBUG_NODE_KEY_H_
#define MACHINA_CORE_DEBUG_DEBUG_NODE_KEY_H_

#include <string>

#include "machina/core/platform/types.h"

namespace machina {

// Encapsulates debug information for a node that was observed.
struct DebugNodeKey {
  static const char* const kMetadataFilePrefix;
  static const char* const kDeviceTag;

  DebugNodeKey(const string& device_name, const string& node_name,
               int32_t output_slot, const string& debug_op,
               const string& io_of_node = "", bool is_input = false,
               int32_t io_index = -1);

  // Converts a device name string to a device path string.
  // E.g., /job:localhost/replica:0/task:0/cpu:0 will be converted to
  //   ,job_localhost,replica_0,task_0,cpu_0.
  static const string DeviceNameToDevicePath(const string& device_name);

  bool operator==(const DebugNodeKey& other) const;
  bool operator!=(const DebugNodeKey& other) const;

  const string device_name;
  const string node_name;
  const int32 output_slot;
  const string debug_op;
  const string debug_node_name;
  const string device_path;
  const string io_of_node;
  const bool is_input;
  const int32 io_index;
};

}  // namespace machina

#endif  // MACHINA_CORE_DEBUG_DEBUG_NODE_KEY_H_
