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

#ifndef MACHINA_LITE_DELEGATES_GPU_COMMON_TASK_BUFFER_DESC_H_
#define MACHINA_LITE_DELEGATES_GPU_COMMON_TASK_BUFFER_DESC_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "machina/lite/delegates/gpu/common/data_type.h"
#include "machina/lite/delegates/gpu/common/status.h"
#include "machina/lite/delegates/gpu/common/task/gpu_object_desc.h"

namespace tflite {
namespace gpu {

struct BufferDescriptor : public GPUObjectDescriptor {
  DataType element_type;
  int element_size;
  MemoryType memory_type = MemoryType::GLOBAL;
  std::vector<std::string> attributes;

  // optional
  int size = 0;
  std::vector<uint8_t> data;

  BufferDescriptor() = default;
  BufferDescriptor(const BufferDescriptor&) = default;
  BufferDescriptor& operator=(const BufferDescriptor&) = default;
  BufferDescriptor(BufferDescriptor&& desc) = default;
  BufferDescriptor& operator=(BufferDescriptor&& desc) = default;

  absl::Status PerformSelector(const GpuInfo& gpu_info,
                               absl::string_view selector,
                               const std::vector<std::string>& args,
                               const std::vector<std::string>& template_args,
                               std::string* result) const override;

  GPUResources GetGPUResources(const GpuInfo& gpu_info) const override;
  absl::Status PerformReadSelector(const GpuInfo& gpu_info,
                                   const std::vector<std::string>& args,
                                   std::string* result) const;
  absl::Status PerformWriteSelector(const GpuInfo& gpu_info,
                                    const std::vector<std::string>& args,
                                    std::string* result) const;
  absl::Status PerformGetPtrSelector(
      const std::vector<std::string>& args,
      const std::vector<std::string>& template_args, std::string* result) const;

  void Release() override;

  uint64_t GetSizeInBytes() const override { return data.size(); };
};

}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_COMMON_TASK_BUFFER_DESC_H_
