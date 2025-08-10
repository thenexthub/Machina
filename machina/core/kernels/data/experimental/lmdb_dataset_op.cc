/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
#include "machina/core/kernels/data/experimental/lmdb_dataset_op.h"

#include <sys/stat.h>

#include "machina/core/framework/dataset.h"

namespace machina {
namespace data {
namespace experimental {

/* static */ constexpr const char* const LMDBDatasetOp::kDatasetType;
/* static */ constexpr const char* const LMDBDatasetOp::kFileNames;
/* static */ constexpr const char* const LMDBDatasetOp::kOutputTypes;
/* static */ constexpr const char* const LMDBDatasetOp::kOutputShapes;

void LMDBDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase** output) {
  OP_REQUIRES(
      ctx, false,
      errors::Unimplemented(
          "LMDB support is removed from TensorFlow. This API will be deleted "
          "in the next TensorFlow release. If you need LMDB support, please "
          "file a GitHub issue."));
}

namespace {

REGISTER_KERNEL_BUILDER(Name("LMDBDataset").Device(DEVICE_CPU), LMDBDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalLMDBDataset").Device(DEVICE_CPU),
                        LMDBDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace machina
