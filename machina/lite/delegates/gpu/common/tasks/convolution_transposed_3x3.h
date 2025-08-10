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

#ifndef MACHINA_LITE_DELEGATES_GPU_COMMON_TASKS_CONVOLUTION_TRANSPOSED_3X3_H_
#define MACHINA_LITE_DELEGATES_GPU_COMMON_TASKS_CONVOLUTION_TRANSPOSED_3X3_H_

#include <string>
#include <vector>

#include "machina/lite/delegates/gpu/common/data_type.h"
#include "machina/lite/delegates/gpu/common/operations.h"
#include "machina/lite/delegates/gpu/common/shape.h"
#include "machina/lite/delegates/gpu/common/status.h"
#include "machina/lite/delegates/gpu/common/task/buffer_desc.h"
#include "machina/lite/delegates/gpu/common/task/gpu_operation.h"
#include "machina/lite/delegates/gpu/common/task/tensor_desc.h"
#include "machina/lite/delegates/gpu/common/task/weights_conversion.h"
#include "machina/lite/delegates/gpu/common/task/weights_layout.h"
#include "machina/lite/delegates/gpu/common/tensor.h"
#include "machina/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

class ConvolutionTransposed3x3 : public GPUOperation {
 public:
  ConvolutionTransposed3x3() = default;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override;
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;

  // Move only
  ConvolutionTransposed3x3(ConvolutionTransposed3x3&& operation) = default;
  ConvolutionTransposed3x3& operator=(ConvolutionTransposed3x3&& operation) =
      default;
  ConvolutionTransposed3x3(const ConvolutionTransposed3x3&) = delete;
  ConvolutionTransposed3x3& operator=(const ConvolutionTransposed3x3&) = delete;

  WeightsDescription GetWeightsDescription() const {
    WeightsDescription desc;
    desc.type = DeduceDataTypeFromPrecision(definition_.precision);
    desc.layout = weights_layout_;
    desc.spatial_remap = GetSpatialWeightsRemap();
    return desc;
  }

  enum class WeightsUploadType {
    LOCAL_MEM_ASYNC,
    LOCAL_MEM_BY_THREADS,
    GLOBAL_MEM,
    CONSTANT_MEM,
  };

 private:
  ConvolutionTransposed3x3(const OperationDef& definition,
                           const GpuInfo& gpu_info, int2 padding);
  friend ConvolutionTransposed3x3 CreateConvolutionTransposed3x3(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const ConvolutionTransposedAttributes& attr);
  friend ConvolutionTransposed3x3 CreateConvolutionTransposed3x3DynamicWeights(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const ConvolutionTransposedAttributes& attr);

  void UploadWeights(
      const tflite::gpu::Tensor<OHWI, DataType::FLOAT32>& weights);

  std::vector<int> GetSpatialWeightsRemap() const;

  std::string GenerateConvolutionTransposedCode(
      const GpuInfo& gpu_info, const OperationDef& op_def,
      ConvolutionTransposed3x3::WeightsUploadType weights_upload_type,
      int2 padding, int3 work_group_launch_order);

  int2 padding_;
  WeightsUploadType weights_upload_type_;
  WeightsLayout weights_layout_;
};

bool IsConvolutionTransposed3x3Supported(
    const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr);

ConvolutionTransposed3x3 CreateConvolutionTransposed3x3(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr);

ConvolutionTransposed3x3 CreateConvolutionTransposed3x3DynamicWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr);

}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_COMMON_TASKS_CONVOLUTION_TRANSPOSED_3X3_H_
