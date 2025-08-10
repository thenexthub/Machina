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

#ifndef MACHINA_CORE_PLATFORM_ERROR_PAYLOADS_H_
#define MACHINA_CORE_PLATFORM_ERROR_PAYLOADS_H_

#include "absl/status/status.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/protobuf/core_platform_payloads.pb.h"
// This file contains macros and payload keys for the error counter in
// EagerClient.

namespace tsl {

// Proto: machina::core::platform::ErrorSourceProto
// Location: machina/core/protobuf/core_platform_payloads.proto
// Usage: Payload key for recording the error raised source. Payload value is
// retrieved to update counter in
// machina/core/distributed_runtime/rpc/eager/grpc_eager_client.cc.
constexpr char kErrorSource[] =
    "type.googleapis.com/machina.core.platform.ErrorSourceProto";

// Set payload when status is not ok and ErrorSource payload hasn't been set.
// The code below will be used at every place where we would like to catch
// the error for the error counter in EagerClient.

void OkOrSetErrorCounterPayload(
    const machina::core::platform::ErrorSourceProto::ErrorSource&
        error_source,
    absl::Status& status);
}  // namespace tsl

namespace machina {
using tsl::kErrorSource;                // NOLINT
using tsl::OkOrSetErrorCounterPayload;  // NOLINT
}  // namespace machina

#endif  // MACHINA_CORE_PLATFORM_ERROR_PAYLOADS_H_
