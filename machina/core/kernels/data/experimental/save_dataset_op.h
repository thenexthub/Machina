/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#ifndef MACHINA_CORE_KERNELS_DATA_EXPERIMENTAL_SAVE_DATASET_OP_H_
#define MACHINA_CORE_KERNELS_DATA_EXPERIMENTAL_SAVE_DATASET_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "machina/core/data/captured_function.h"
#include "machina/core/framework/dataset.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/kernels/data/iterator_ops.h"
#include "machina/core/platform/thread_annotations.h"

namespace machina {
namespace data {
namespace experimental {

// An operation that can save a dataset to one or more files.
class SaveDatasetOp : public HybridAsyncOpKernel {
 public:
  static constexpr const char* const kCompression = "compression";
  static constexpr const char* const kPath = "path";
  static constexpr const char* const kShardFunc = "shard_func";
  static constexpr const char* const kShardFuncOtherArgs =
      "shard_func_other_args";
  static constexpr const char* const kUseShardFunc = "use_shard_func";

  explicit SaveDatasetOp(OpKernelConstruction* ctx);

  absl::Status DoCompute(OpKernelContext* ctx) override;

 private:
  static constexpr const int kFileFormatVersion = 2;

  absl::Status ConsumeElement();

  absl::Status GetShardIndex(IteratorContext* ctx,
                             InstantiatedCapturedFunction* function,
                             const std::vector<Tensor>& element,
                             int64_t* shard_index);

  absl::Status WriteData(OpKernelContext* ctx, DatasetBase* dataset,
                         std::unique_ptr<CapturedFunction> captured_func,
                         const std::string& run_dir, uint64* num_elements);

  absl::Status WriteMetadataFile(Env* env, const std::string& path,
                                 uint64 run_id,
                                 const DataTypeVector& output_dtypes,
                                 uint64 num_elements, bool finalized);

  bool use_shard_func_;
  std::string compression_;
  std::shared_ptr<FunctionMetadata> func_metadata_;
};

// An operation that can save a dataset to one or more files. This
// version of the implementation subclasses from UnaryDatasetOpKernel to align
// the implementation of save with that of the other tf.data transformations.
class SaveDatasetV2Op : public UnaryDatasetOpKernel {
 public:
  static constexpr const char* const kInputDataset = "input_dataset";
  static constexpr const char* const kPath = "path";
  static constexpr const char* const kCompression = "compression";

  static constexpr const char* const kDatasetType = "SaveV2";
  static constexpr const char* const kOutputTypes = "output_types";
  static constexpr const char* const kOutputShapes = "output_shapes";

  static constexpr const char* const kShardFunc = "shard_func";
  static constexpr const char* const kShardFuncOtherArgs =
      "shard_func_other_args";
  static constexpr const char* const kUseShardFunc = "use_shard_func";
  static constexpr const char* const kShardFuncTarguments = "Tshard_func_args";

  explicit SaveDatasetV2Op(OpKernelConstruction* ctx);

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;

  static constexpr const int kFileFormatVersion = 2;

  tstring path_;
  std::string compression_;
  std::unique_ptr<CapturedFunction> shard_func_;
  bool use_shard_func_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  std::shared_ptr<FunctionMetadata> func_metadata_;
  std::string writer_prefix_;
};

}  // namespace experimental
}  // namespace data
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_DATA_EXPERIMENTAL_SAVE_DATASET_OP_H_
