/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#ifndef MACHINA_LITE_DELEGATES_GPU_COMMON_TASK_QCOM_THIN_FILTER_DESC_H_
#define MACHINA_LITE_DELEGATES_GPU_COMMON_TASK_QCOM_THIN_FILTER_DESC_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "machina/lite/delegates/gpu/common/status.h"
#include "machina/lite/delegates/gpu/common/task/gpu_object_desc.h"

namespace tflite {
namespace gpu {

struct QcomThinFilterDescriptor : public GPUObjectDescriptor {
  int kernel_size_x;
  int kernel_size_y;
  std::vector<uint8_t> data;

  QcomThinFilterDescriptor() = default;
  QcomThinFilterDescriptor(const QcomThinFilterDescriptor&) = default;
  QcomThinFilterDescriptor& operator=(const QcomThinFilterDescriptor&) =
      default;
  QcomThinFilterDescriptor(QcomThinFilterDescriptor&& desc) = default;
  QcomThinFilterDescriptor& operator=(QcomThinFilterDescriptor&& desc) =
      default;

  absl::Status PerformSelector(const GpuInfo& gpu_info,
                               absl::string_view selector,
                               const std::vector<std::string>& args,
                               const std::vector<std::string>& template_args,
                               std::string* result) const override;

  GPUResources GetGPUResources(const GpuInfo& gpu_info) const override;

  void Release() override { data.clear(); }
};

}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_COMMON_TASK_QCOM_THIN_FILTER_DESC_H_
