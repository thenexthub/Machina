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
#include "machina/core/kernels/collective_nccl_all_to_all.h"

#if GOOGLE_CUDA || MACHINA_USE_ROCM

#include "machina/core/common_runtime/collective_util.h"
#include "machina/core/nccl/nccl_manager.h"
#include "machina/core/profiler/lib/traceme.h"

namespace machina {

void NcclAllToAll::Run(StatusCallback done) {
  col_ctx_->nccl_communicator->Enqueue(col_ctx_, std::move(done));
}

REGISTER_COLLECTIVE(NcclAllToAll, NcclAllToAll);

}  // namespace machina

#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM
