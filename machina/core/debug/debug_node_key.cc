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

#include "machina/core/debug/debug_node_key.h"

#include <cstdint>

#include "machina/core/lib/strings/str_util.h"
#include "machina/core/lib/strings/strcat.h"

namespace machina {

const char* const DebugNodeKey::kMetadataFilePrefix = "_tfdbg_";

const char* const DebugNodeKey::kDeviceTag = "device_";

DebugNodeKey::DebugNodeKey(const string& device_name, const string& node_name,
                           const int32_t output_slot, const string& debug_op,
                           const string& io_of_node, const bool is_input,
                           const int32_t io_index)
    : device_name(device_name),
      node_name(node_name),
      output_slot(output_slot),
      debug_op(debug_op),
      debug_node_name(
          strings::StrCat(node_name, ":", output_slot, ":", debug_op)),
      device_path(DeviceNameToDevicePath(device_name)),
      io_of_node(io_of_node),
      is_input(is_input),
      io_index(io_index) {}

bool DebugNodeKey::operator==(const DebugNodeKey& other) const {
  return (device_name == other.device_name && node_name == other.node_name &&
          output_slot == other.output_slot && debug_op == other.debug_op &&
          io_of_node == other.io_of_node && is_input == other.is_input &&
          io_index == other.io_index);
}

bool DebugNodeKey::operator!=(const DebugNodeKey& other) const {
  return !((*this) == other);
}

const string DebugNodeKey::DeviceNameToDevicePath(const string& device_name) {
  return strings::StrCat(
      kMetadataFilePrefix, kDeviceTag,
      str_util::StringReplace(
          str_util::StringReplace(device_name, ":", "_", true), "/", ",",
          true));
}

}  // namespace machina
