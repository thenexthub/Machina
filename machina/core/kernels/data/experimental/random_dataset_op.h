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
#ifndef MACHINA_CORE_KERNELS_DATA_EXPERIMENTAL_RANDOM_DATASET_OP_H_
#define MACHINA_CORE_KERNELS_DATA_EXPERIMENTAL_RANDOM_DATASET_OP_H_

#include "machina/core/framework/dataset.h"

namespace machina {
namespace data {
namespace experimental {

// See machina/core/api_def/base_api/api_def_RandomDataset.pbtxt for the
// API definition that corresponds to this kernel.
class RandomDatasetOp : public DatasetOpKernel {
 public:
  // Names of op parameters, public so that they can be accessed by test cases.
  // Make sure that these are kept in sync with the REGISTER_OP call in
  // machina/core/ops/experimental_dataset_ops.cc
  static constexpr const char* const kDatasetType = "Random";
  static constexpr const char* const kSeed = "seed";
  static constexpr const char* const kSeed2 = "seed2";
  static constexpr const char* const kOutputTypes = "output_types";
  static constexpr const char* const kOutputShapes = "output_shapes";
  static constexpr const char* const kRerandomizeEachIteration =
      "rerandomize_each_iteration";

  explicit RandomDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override;

 private:
  class Dataset;
  int32_t op_version_;
  bool rerandomize_each_iteration_ = false;
};

}  // namespace experimental
}  // namespace data
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_DATA_EXPERIMENTAL_RANDOM_DATASET_OP_H_
