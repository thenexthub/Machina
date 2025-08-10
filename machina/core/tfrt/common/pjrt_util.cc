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
#include "machina/core/tfrt/common/pjrt_util.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "machina/xla/pjrt/pjrt_client.h"
#include "machina/core/framework/resource_mgr.h"
#include "machina/core/framework/types.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/refcount.h"
#include "machina/core/platform/status.h"
#include "machina/core/tfrt/common/global_state.h"
#include "machina/core/tfrt/common/pjrt_state.h"
#include "tsl/platform/errors.h"

namespace machina {

absl::Status SetPjRtClientInTFGlobalResourceManager(
    const DeviceType& device_type, std::unique_ptr<xla::PjRtClient> client) {
  ResourceMgr* rmgr = tfrt_global::GetTFGlobalResourceMgr();
  PjRtState* pjrt_state;
  TF_RETURN_IF_ERROR(rmgr->LookupOrCreate<PjRtState>(
      rmgr->default_container(), kPjRtStateResourceName, &pjrt_state,
      [&](PjRtState** ret) {
        *ret = PjRtState::Create();
        return absl::OkStatus();
      }));
  core::ScopedUnref pjrt_state_ref(pjrt_state);
  if (client == nullptr) {
    return errors::InvalidArgument("PJRT client is nullptr.");
  }
  TF_RETURN_IF_ERROR(pjrt_state->SetPjRtClient(device_type, std::move(client)));
  return absl::OkStatus();
}

absl::StatusOr<xla::PjRtClient*> GetPjRtClient(const DeviceType& device_type) {
  ResourceMgr* rmgr = tfrt_global::GetTFGlobalResourceMgr();
  PjRtState* pjrt_state;
  TF_RETURN_IF_ERROR(rmgr->LookupOrCreate<PjRtState>(
      rmgr->default_container(), kPjRtStateResourceName, &pjrt_state,
      [&](PjRtState** ret) {
        *ret = PjRtState::Create();
        return absl::OkStatus();
      }));
  core::ScopedUnref pjrt_state_ref(pjrt_state);
  return pjrt_state->GetPjRtClient(device_type);
}

absl::Status SetPjRtGpuClientCreationInfoInTFGlobalResourceManager(
    std::unique_ptr<PjRtGpuClientCreationInfo> info) {
  ResourceMgr* rmgr = tfrt_global::GetTFGlobalResourceMgr();
  PjRtState* pjrt_state;
  TF_RETURN_IF_ERROR(rmgr->LookupOrCreate<PjRtState>(
      rmgr->default_container(), kPjRtStateResourceName, &pjrt_state,
      [&](PjRtState** ret) {
        *ret = PjRtState::Create();
        return absl::OkStatus();
      }));
  core::ScopedUnref pjrt_state_ref(pjrt_state);
  if (info == nullptr) {
    return absl::InvalidArgumentError("PJRT client creation info is nullptr.");
  }
  TF_RETURN_IF_ERROR(pjrt_state->SetPjRtGpuClientCreationInfo(std::move(info)));
  return absl::OkStatus();
}

absl::StatusOr<PjRtGpuClientCreationInfo*> GetPjRtGpuClientCreationInfo() {
  ResourceMgr* rmgr = tfrt_global::GetTFGlobalResourceMgr();
  PjRtState* pjrt_state;
  TF_RETURN_IF_ERROR(rmgr->LookupOrCreate<PjRtState>(
      rmgr->default_container(), kPjRtStateResourceName, &pjrt_state,
      [&](PjRtState** ret) {
        *ret = PjRtState::Create();
        return absl::OkStatus();
      }));
  core::ScopedUnref pjrt_state_ref(pjrt_state);
  return pjrt_state->GetPjRtGpuClientCreationInfo();
}
}  // namespace machina
