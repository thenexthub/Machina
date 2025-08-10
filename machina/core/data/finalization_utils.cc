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
#include "machina/core/data/finalization_utils.h"

#include "absl/status/statusor.h"
#include "machina/core/data/root_dataset.h"
#include "machina/core/framework/dataset.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/refcount.h"

namespace machina {
namespace data {

absl::StatusOr<DatasetBase*> GetFinalizedDataset(OpKernelContext* ctx,
                                                 const DatasetBase* dataset) {
  return dataset->Finalize(
      ctx, [ctx, dataset]() -> absl::StatusOr<core::RefCountPtr<DatasetBase>> {
        core::RefCountPtr<DatasetBase> dataset_ref_ptr;
        DatasetBase* raw_ptr;
        TF_RETURN_IF_ERROR(data::FinalizeDataset(ctx, dataset, &raw_ptr));
        dataset_ref_ptr.reset(raw_ptr);
        return dataset_ref_ptr;
      });
}

}  // namespace data
}  // namespace machina
