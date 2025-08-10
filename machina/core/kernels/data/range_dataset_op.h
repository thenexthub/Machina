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
#ifndef MACHINA_CORE_KERNELS_DATA_RANGE_DATASET_OP_H_
#define MACHINA_CORE_KERNELS_DATA_RANGE_DATASET_OP_H_

#include "machina/core/framework/dataset.h"

namespace machina {
namespace data {

class RangeDatasetOp : public DatasetOpKernel {
 public:
  static constexpr const char* const kDatasetType = "Range";
  static constexpr const char* const kStart = "start";
  static constexpr const char* const kStop = "stop";
  static constexpr const char* const kStep = "step";
  static constexpr const char* const kOutputTypes = "output_types";
  static constexpr const char* const kOutputShapes = "output_shapes";
  static constexpr const char* const kReplicateOnSplit = "replicate_on_split";

  explicit RangeDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override;

 private:
  class Dataset;
  class RangeSplitProvider;
  DataTypeVector output_types_;
  bool replicate_on_split_ = false;
};

}  // namespace data
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_DATA_RANGE_DATASET_OP_H_
