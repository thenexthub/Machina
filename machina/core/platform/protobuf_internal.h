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

#ifndef MACHINA_CORE_PLATFORM_PROTOBUF_INTERNAL_H_
#define MACHINA_CORE_PLATFORM_PROTOBUF_INTERNAL_H_

#include "google/protobuf/any.pb.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/platform.h"
#include "machina/core/platform/protobuf.h"
#include "machina/core/platform/types.h"

namespace machina {

// Utility for parsing an Any value with full or lite protos.
template <class T>
absl::Status ParseAny(const google::protobuf::Any& any, T* message,
                      const string& type_name) {
  CHECK_EQ(type_name, message->GetTypeName());
  if (!any.Is<T>()) {
    return errors::FailedPrecondition(
        "Expected Any type_url for: ", message->GetTypeName(),
        ". Got: ", string(any.type_url().data(), any.type_url().size()), ".");
  }
  if (!any.UnpackTo(message)) {
    return errors::FailedPrecondition("Failed to unpack: ", any.DebugString());
  }
  return absl::OkStatus();
}

}  // namespace machina

#endif  // MACHINA_CORE_PLATFORM_PROTOBUF_INTERNAL_H_
