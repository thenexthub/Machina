/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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
#include "machina/core/kernels/collective_nccl.h"

#if GOOGLE_CUDA || MACHINA_USE_ROCM

#include "machina/core/common_runtime/collective_util.h"
#include "machina/core/nccl/nccl_manager.h"
#include "machina/core/profiler/lib/traceme.h"

namespace machina {

NcclBase::NcclBase(CollectiveType type, const string& name)
    : type_(type), name_(name), col_ctx_(nullptr), col_params_(nullptr) {}

Status NcclBase::InitializeCollectiveParams(CollectiveParams* col_params) {
  if (type_ != col_params->instance.type) {
    return errors::Internal("Expected initialized type ", type_,
                            " to match type in CollectiveParams ",
                            col_params->instance.type);
  }

  const char* expected_name;
  switch (type_) {
    case REDUCTION_COLLECTIVE:
      expected_name = "NcclReduce";
      break;
    case BROADCAST_COLLECTIVE:
      expected_name = "NcclBroadcast";
      break;
    case GATHER_COLLECTIVE:
      expected_name = "NcclGather";
      break;
    case REDUCE_SCATTER_COLLECTIVE:
      expected_name = "NcclReduceScatter";
      break;
    case ALL_TO_ALL_COLLECTIVE:
      expected_name = "NcclAllToAll";
      break;
    default:
      return errors::Internal("Unexpected CollectiveType ", type_);
  }

  if (expected_name != col_params->instance.impl_details.collective_name) {
    return errors::Internal("Unexpected combination of collective type ",
                            col_params->instance.type, " and collective name ",
                            col_params->instance.impl_details.collective_name,
                            ", expected name ", expected_name);
  }

  return OkStatus();
}

Status NcclBase::InitializeCollectiveContext(
    std::shared_ptr<CollectiveContext> col_ctx) {
  col_ctx_ = col_ctx;
  col_params_ = col_ctx->col_params.get();
  return collective_util::InitializeDeviceAndLocality(
      col_ctx->dev_mgr, col_ctx->device_name, &col_ctx->device,
      &col_ctx->device_locality);
}

}  // namespace machina

#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM
