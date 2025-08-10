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

#ifndef MACHINA_LITE_DELEGATES_GPU_COMMON_TASKS_SOFTMAX1X1_H_
#define MACHINA_LITE_DELEGATES_GPU_COMMON_TASKS_SOFTMAX1X1_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "machina/lite/delegates/gpu/common/task/gpu_operation.h"
#include "machina/lite/delegates/gpu/common/task/tensor_desc.h"

namespace tflite {
namespace gpu {

class Softmax1x1 : public GPUOperation {
 public:
  Softmax1x1() = default;
  Softmax1x1(const OperationDef& definition, const GpuInfo& gpu_info,
             const BHWC& shape);
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override {
    work_groups->push_back(work_group_size_);
  }
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;

  // Move only
  Softmax1x1(Softmax1x1&& kernel);
  Softmax1x1& operator=(Softmax1x1&& kernel);
  Softmax1x1(const Softmax1x1&) = delete;
  Softmax1x1& operator=(const Softmax1x1&) = delete;

  friend Softmax1x1 CreateSoftmax1x1();

 private:
  std::string GetSoftmaxKernelCode(const OperationDef& op_def);
};

Softmax1x1 CreateSoftmax1x1(const OperationDef& definition,
                            const GpuInfo& gpu_info, const BHWC& shape);

}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_COMMON_TASKS_SOFTMAX1X1_H_
