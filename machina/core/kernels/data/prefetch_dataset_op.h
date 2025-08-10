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

#ifndef MACHINA_CORE_KERNELS_DATA_PREFETCH_DATASET_OP_H_
#define MACHINA_CORE_KERNELS_DATA_PREFETCH_DATASET_OP_H_

#include "machina/core/framework/dataset.h"
#include "machina/core/framework/model.h"
#include "machina/core/kernels/data/prefetch_autotuner.h"

namespace machina {
namespace data {

class PrefetchDatasetOp : public UnaryDatasetOpKernel {
 public:
  static constexpr const char* const kDatasetType = "Prefetch";
  static constexpr const char* const kInputDataset = "input_dataset";
  static constexpr const char* const kBufferSize = model::kBufferSize;
  static constexpr const char* const kOutputTypes = "output_types";
  static constexpr const char* const kOutputShapes = "output_shapes";
  static constexpr const char* const kSlackPeriod = "slack_period";
  static constexpr const char* const kLegacyAutotune = "legacy_autotune";
  static constexpr const char* const kBufferSizeMin = "buffer_size_min";

  explicit PrefetchDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;
  int64_t slack_period_ = 0;
  bool legacy_autotune_ = true;
  int64_t buffer_size_min_ = 0;
};

}  // namespace data
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_DATA_PREFETCH_DATASET_OP_H_
