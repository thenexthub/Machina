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

#ifndef MACHINA_LITE_DELEGATES_GPU_CL_GPU_OBJECT_H_
#define MACHINA_LITE_DELEGATES_GPU_CL_GPU_OBJECT_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "machina/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "machina/lite/delegates/gpu/common/access_type.h"
#include "machina/lite/delegates/gpu/common/data_type.h"
#include "machina/lite/delegates/gpu/common/status.h"
#include "machina/lite/delegates/gpu/common/task/gpu_object_desc.h"

namespace tflite {
namespace gpu {
namespace cl {

struct GPUResourcesWithValue {
  GenericGPUResourcesWithValue generic;

  std::vector<std::pair<std::string, cl_mem>> buffers;
  std::vector<std::pair<std::string, cl_mem>> images2d;
  std::vector<std::pair<std::string, cl_mem>> image2d_arrays;
  std::vector<std::pair<std::string, cl_mem>> images3d;
  std::vector<std::pair<std::string, cl_mem>> image_buffers;
  std::vector<std::pair<std::string, cl_mem>> custom_memories;

  void AddFloat(const std::string& name, float value) {
    generic.AddFloat(name, value);
  }
  void AddInt(const std::string& name, int value) {
    generic.AddInt(name, value);
  }
};

class GPUObject {
 public:
  GPUObject() = default;
  // Move only
  GPUObject(GPUObject&& obj_desc) = default;
  GPUObject& operator=(GPUObject&& obj_desc) = default;
  GPUObject(const GPUObject&) = delete;
  GPUObject& operator=(const GPUObject&) = delete;
  virtual ~GPUObject() = default;
  virtual absl::Status GetGPUResources(
      const GPUObjectDescriptor* obj_ptr,
      GPUResourcesWithValue* resources) const = 0;
};

using GPUObjectPtr = std::unique_ptr<GPUObject>;

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_CL_GPU_OBJECT_H_
