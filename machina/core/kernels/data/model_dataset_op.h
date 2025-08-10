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
#ifndef MACHINA_CORE_KERNELS_DATA_MODEL_DATASET_OP_H_
#define MACHINA_CORE_KERNELS_DATA_MODEL_DATASET_OP_H_

#include "machina/core/platform/platform.h"

// On mobile we do not provide model dataset op because not all of its
// dependencies are available there. The op is replaced with a no-op.
#if !defined(IS_MOBILE_PLATFORM)
#include "machina/core/framework/dataset.h"
#include "machina/core/framework/model.h"

namespace machina {
namespace data {

class ModelDatasetOp : public UnaryDatasetOpKernel {
 public:
  static constexpr const char* const kDatasetType = "ModelDataset";
  static constexpr const char* const kDatasetOp = "ModelDatasetOp";
  static constexpr const char* const kAlgorithm = "algorithm";
  static constexpr const char* const kCpuBudget = "cpu_budget";
  static constexpr const char* const kRamBudget = "ram_budget";

  // Executes the logic of the ModelDatasetOp directly (as opposed to through
  // executing the ModelDatasetOp op kernel).
  static void MakeDatasetFromOptions(OpKernelContext* ctx, DatasetBase* input,
                                     model::AutotuneAlgorithm algorithm,
                                     int64_t cpu_budget, int64_t ram_budget,
                                     DatasetBase** output);

  explicit ModelDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;

  model::AutotuneAlgorithm algorithm_;
  int64_t cpu_budget_;
  int64_t ram_budget_;
};

}  // namespace data
}  // namespace machina
#else  // !IS_MOBILE_PLATFORM
#include "machina/core/framework/dataset.h"

namespace machina {
namespace data {

class ModelDatasetOp : public UnaryDatasetOpKernel {
 public:
  // Creates and returns a ModelDatasetOp::Dataset in output, given the
  // input, algorithm, cpu_budget and ram_budget parameters. This method is used
  // to create the dataset without explicitly using the ModelDatasetOp.
  static void MakeDatasetFromOptions(OpKernelContext* ctx, DatasetBase* input,
                                     model::AutotuneAlgorithm algorithm,
                                     bool cpu_budget, bool ram_budget,
                                     DatasetBase** output);

  explicit ModelDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;
};

}  // namespace data
}  // namespace machina
#endif  // !IS_MOBILE_PLATFORM

#endif  // MACHINA_CORE_KERNELS_DATA_MODEL_DATASET_OP_H_
