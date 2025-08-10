/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#include "machina/core/kernels/data/experimental/random_access_ops.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "machina/core/data/finalization_utils.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/statusor.h"

namespace machina {
namespace data {
namespace experimental {

absl::Status GetElementAtIndexOp::DoCompute(OpKernelContext* ctx) {
  DatasetBase* dataset;
  TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(ctx->input(0), &dataset));

  DatasetBase* finalized_dataset;
  TF_ASSIGN_OR_RETURN(finalized_dataset, GetFinalizedDataset(ctx, dataset));

  int64 index = 0;
  TF_RETURN_IF_ERROR(ParseScalarArgument<int64_t>(ctx, "index", &index));

  std::vector<Tensor> components;

  TF_RETURN_IF_ERROR(finalized_dataset->Get(ctx, index, &components));
  TF_RETURN_IF_ERROR(VerifyTypesMatch(output_types_, components));
  TF_RETURN_IF_ERROR(VerifyShapesCompatible(output_shapes_, components));

  for (int i = 0; i < components.size(); ++i) {
    ctx->set_output(i, components[i]);
  }
  return absl::OkStatus();
}

namespace {

REGISTER_KERNEL_BUILDER(Name("GetElementAtIndex").Device(DEVICE_CPU),
                        GetElementAtIndexOp);

}

}  // namespace experimental
}  // namespace data
}  // namespace machina
