/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#include "machina/core/platform/error_payloads.h"

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "machina/core/protobuf/core_platform_payloads.pb.h"

namespace tsl {

using ::machina::core::platform::ErrorSourceProto;

void OkOrSetErrorCounterPayload(
    const ErrorSourceProto::ErrorSource& error_source, absl::Status& status) {
  if (!status.ok() &&
      !status.GetPayload(machina::kErrorSource).has_value()) {
    ErrorSourceProto error_source_proto;
    error_source_proto.set_error_source(error_source);
    status.SetPayload(machina::kErrorSource,
                      absl::Cord(error_source_proto.SerializeAsString()));
  }
}

}  // namespace tsl
