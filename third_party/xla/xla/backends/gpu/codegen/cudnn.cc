/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#include "machina/xla/backends/gpu/codegen/cudnn.h"

#include <memory>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "machina/xla/backends/gpu/codegen/fusion_emitter.h"
#include "machina/xla/backends/gpu/runtime/cudnn_thunk.h"
#include "machina/xla/backends/gpu/runtime/thunk.h"
#include "machina/xla/codegen/emitters/computation_fingerprint.h"
#include "machina/xla/codegen/emitters/kernel_arguments.h"
#include "machina/xla/hlo/ir/hlo_instructions.h"
#include "machina/xla/service/gpu/gpu_constants.h"
#include "machina/xla/service/gpu/ir_emitter_context.h"
#include "machina/xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

absl::StatusOr<FusionEmissionResult> CuDnnFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  VLOG(3) << fusion.ToString();

  TF_ASSIGN_OR_RETURN(
      auto kernel_arguments,
      emitters::KernelArguments::Create(ir_emitter_context.buffer_assignment(),
                                        GetDefaultBufferAlignment(), &fusion));
  FusionEmissionResult result;
  result.thunks.emplace_back(std::make_unique<CuDnnThunk>(
      emitters::GetComputationFingerprint(
          fusion.fused_instructions_computation(), {}),
      Thunk::ThunkInfo::WithProfileAnnotation(&fusion),
      kernel_arguments.GetArgumentBufferSlices(),
      kernel_arguments.GetArgumentOutputFlags()));
  return result;
}

}  // namespace gpu
}  // namespace xla
