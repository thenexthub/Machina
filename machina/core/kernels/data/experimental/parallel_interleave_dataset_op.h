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
#ifndef MACHINA_CORE_KERNELS_DATA_EXPERIMENTAL_PARALLEL_INTERLEAVE_DATASET_OP_H_
#define MACHINA_CORE_KERNELS_DATA_EXPERIMENTAL_PARALLEL_INTERLEAVE_DATASET_OP_H_

#include "machina/core/data/captured_function.h"
#include "machina/core/data/dataset_utils.h"
#include "machina/core/framework/dataset.h"

namespace machina {
namespace data {
namespace experimental {

// See documentation in ../../ops/experimental_dataset_ops.cc for a high-level
// description of the following op.

class ParallelInterleaveDatasetOp : public UnaryDatasetOpKernel {
 public:
  static constexpr const char* const kDatasetType = "LegacyParallelInterleave";
  static constexpr const char* const kInputDataset = "input_dataset";
  static constexpr const char* const kOtherArguments = "other_arguments";
  static constexpr const char* const kCycleLength = "cycle_length";
  static constexpr const char* const kBlockLength = "block_length";
  static constexpr const char* const kDeterministic = "deterministic";
  static constexpr const char* const kSloppy = "sloppy";
  static constexpr const char* const kBufferOutputElements =
      "buffer_output_elements";
  static constexpr const char* const kPrefetchInputElements =
      "prefetch_input_elements";
  static constexpr const char* const kFunc = "f";
  static constexpr const char* const kTarguments = "Targuments";
  static constexpr const char* const kOutputTypes = "output_types";
  static constexpr const char* const kOutputShapes = "output_shapes";

  explicit ParallelInterleaveDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;
  const int op_version_;

  std::shared_ptr<FunctionMetadata> func_metadata_ = nullptr;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  DeterminismPolicy deterministic_;
};

}  // namespace experimental
}  // namespace data
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_DATA_EXPERIMENTAL_PARALLEL_INTERLEAVE_DATASET_OP_H_
