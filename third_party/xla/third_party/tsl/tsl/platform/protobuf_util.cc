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

#include <string>

#include "tsl/platform/protobuf.h"

namespace tsl {

bool ParseProtoUnlimited(protobuf::MessageLite* proto,
                         const string& serialized) {
  return proto->ParseFromString(serialized);
}

bool ParseProtoUnlimited(protobuf::MessageLite* proto, const void* serialized,
                         size_t size) {
  return proto->ParseFromArray(serialized, size);
}

std::string LegacyUnredactedDebugString(const tsl::protobuf::Message& message) {
  std::string debug_string;
  tsl::protobuf::TextFormat::Printer printer;
  printer.SetExpandAny(true);

  printer.PrintToString(message, &debug_string);
  return debug_string;
}

std::string LegacyUnredactedDebugString(
    const tsl::protobuf::MessageLite& message) {
  return message.DebugString();
}

std::string LegacyUnredactedShortDebugString(
    const tsl::protobuf::Message& message) {
  std::string debug_string;
  tsl::protobuf::TextFormat::Printer printer;
  printer.SetSingleLineMode(true);
  printer.SetExpandAny(true);

  printer.PrintToString(message, &debug_string);
  if (!debug_string.empty() && debug_string.back() == ' ') {
    debug_string.pop_back();
  }
  return debug_string;
}

}  // namespace tsl
