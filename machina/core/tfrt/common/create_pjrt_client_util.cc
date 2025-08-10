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
#include "machina/core/tfrt/common/create_pjrt_client_util.h"

#include <optional>
#include <set>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "machina/xla/pjrt/pjrt_client.h"
#include "machina/core/framework/resource_mgr.h"
#include "machina/core/framework/types.h"
#include "machina/core/platform/refcount.h"
#include "machina/core/tfrt/common/global_state.h"
#include "machina/core/tfrt/common/pjrt_state.h"
#include "tsl/platform/errors.h"

namespace machina {

absl::StatusOr<xla::PjRtClient*> GetOrCreatePjRtClient(
    const DeviceType& device_type,
    std::optional<std::set<int>> allowed_devices) {
  ResourceMgr* rmgr = tfrt_global::GetTFGlobalResourceMgr();
  PjRtState* pjrt_state;
  TF_RETURN_IF_ERROR(rmgr->LookupOrCreate<PjRtState>(
      rmgr->default_container(), kPjRtStateResourceName, &pjrt_state,
      [&](PjRtState** ret) {
        *ret = PjRtState::Create();
        return absl::OkStatus();
      }));
  core::ScopedUnref pjrt_state_ref(pjrt_state);
  return pjrt_state->GetOrCreatePjRtClient(device_type);
}

}  // namespace machina
