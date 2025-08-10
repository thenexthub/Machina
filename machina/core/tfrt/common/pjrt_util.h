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
#ifndef MACHINA_CORE_TFRT_COMMON_PJRT_UTIL_H_
#define MACHINA_CORE_TFRT_COMMON_PJRT_UTIL_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "machina/xla/pjrt/pjrt_client.h"
#include "machina/core/framework/types.h"
#include "machina/core/platform/status.h"
#include "machina/core/tfrt/common/pjrt_state.h"

namespace machina {

// Sets PJRT client for device_type in TFGlobalResourceManager. If a PJRT client
// for this device_type already exists, the existing PJRT client will not be
// destroyed, and will be kept alive in an "unused client" vector. PJRT API
// semantics require the PJRT client to outlive PJRT buffers.
absl::Status SetPjRtClientInTFGlobalResourceManager(
    const DeviceType& device_type, std::unique_ptr<xla::PjRtClient> client);

// Gets (the most recent) PJRT client for device_type from
// TFGlobalResourceManager.
absl::StatusOr<xla::PjRtClient*> GetPjRtClient(const DeviceType& device_type);

absl::Status SetPjRtGpuClientCreationInfoInTFGlobalResourceManager(
    std::unique_ptr<PjRtGpuClientCreationInfo> info);
absl::StatusOr<PjRtGpuClientCreationInfo*> GetPjRtGpuClientCreationInfo();

}  // namespace machina

#endif  // MACHINA_CORE_TFRT_COMMON_PJRT_UTIL_H_
