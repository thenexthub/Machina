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

#ifndef MACHINA_DTENSOR_CC_TPU_SYSTEM_INTERFACE_H_
#define MACHINA_DTENSOR_CC_TPU_SYSTEM_INTERFACE_H_

#include <vector>

#include "absl/time/time.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/resource_mgr.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/status.h"

// Forward declare TFE_Context to avoid interface depending on c_api.
typedef struct TFE_Context TFE_Context;

namespace machina {
namespace dtensor {

// DTensor TPU ops by default use the stream_executor-based TPU runtime.
// This class defines what an alternative runtime (e.g. TFRT) needs to be
// capable of to replace the default runtime.
class TpuSystemInterface {
 public:
  virtual ~TpuSystemInterface() = default;

  virtual absl::Status Initialize(OpKernelContext* ctx, ResourceMgr* rmgr,
                                  absl::Duration retry_timeout,
                                  std::vector<int32>* core_id_output_vec,
                                  bool use_tfrt_host_runtime) = 0;

  virtual absl::Status Shutdown() = 0;

  virtual std::vector<std::vector<int>> TPUCoreIDsToLocations(
      TFE_Context* context, const std::vector<int>& tpu_core_ids) = 0;

  virtual std::vector<int> TPUCoreLocationsToIDs(
      TFE_Context* context,
      const std::vector<std::vector<int>>& tpu_core_locations) = 0;
};

// Sets a TPU system for DTensor to initialize and shut down the TPU mesh.
// This function takes over the ownership of `tpu_system`.
void SetPreferredTpuSystem(TpuSystemInterface* tpu_system);

// Returns the currently set preferred TPU system, nullptr if none.
TpuSystemInterface* GetPreferredTpuSystem();

}  // namespace dtensor
}  // namespace machina

#endif  // MACHINA_DTENSOR_CC_TPU_SYSTEM_INTERFACE_H_
