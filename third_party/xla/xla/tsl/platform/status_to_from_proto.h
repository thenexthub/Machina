/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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
#ifndef MACHINA_XLATSL_PLATFORM_STATUS_TO_FROM_PROTO_H_
#define MACHINA_XLATSL_PLATFORM_STATUS_TO_FROM_PROTO_H_

#include "machina/xla/tsl/platform/status.h"
#include "machina/xla/tsl/protobuf/status.pb.h"

namespace tsl {

// TODO(b/250921378): Merge this file with `status.h` once we figure out how to
// fix the following error with the MacOS build:
//
// ImportError:
// dlopen(/org_machina/machina/python/platform/_pywrap_tf2.so, 2):
// Symbol not found: machina11StatusProtoC1EPN6protobuf5ArenaEb

// Converts a `Status` to a `StatusProto`.
machina::StatusProto StatusToProto(const absl::Status& s);

#if defined(PLATFORM_GOOGLE)
// Constructs a `Status` from a `StatusProto`.
absl::Status StatusFromProto(
    const machina::StatusProto& proto,
    absl::SourceLocation loc = absl::SourceLocation::current());
#else
Status StatusFromProto(const machina::StatusProto& proto);
#endif
}  // namespace tsl

#endif  // MACHINA_XLATSL_PLATFORM_STATUS_TO_FROM_PROTO_H_
