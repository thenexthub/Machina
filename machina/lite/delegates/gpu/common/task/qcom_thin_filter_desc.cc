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

#include "machina/lite/delegates/gpu/common/task/qcom_thin_filter_desc.h"

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace tflite {
namespace gpu {

GPUResources QcomThinFilterDescriptor::GetGPUResources(
    const GpuInfo& gpu_info) const {
  GPUResources resources;
  GPUCustomMemoryDescriptor desc;
  desc.type_name = "__read_only qcom_weight_image_t";
  resources.custom_memories.push_back({"filter", desc});
  return resources;
}

absl::Status QcomThinFilterDescriptor::PerformSelector(
    const GpuInfo& gpu_info, absl::string_view selector,
    const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
  if (selector == "GetHandle" && args.empty()) {
    *result = "filter";
    return absl::OkStatus();
  } else {
    return absl::NotFoundError(absl::StrCat(
        "QcomThinFilterDescriptor don't have selector with name - ", selector));
  }
}

}  // namespace gpu
}  // namespace tflite
