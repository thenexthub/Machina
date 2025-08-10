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

#ifndef MACHINA_LITE_DELEGATES_GPU_GL_WORKGROUPS_IDEAL_WORKGROUP_PICKER_H_
#define MACHINA_LITE_DELEGATES_GPU_GL_WORKGROUPS_IDEAL_WORKGROUP_PICKER_H_

#include "machina/lite/delegates/gpu/common/gpu_info.h"
#include "machina/lite/delegates/gpu/common/operations.h"
#include "machina/lite/delegates/gpu/common/shape.h"
#include "machina/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace gl {

// Picks up the ideal workgroup size for the given convolution case.
// Ideal workgroup gives top 10% of the possible performance for the given case.
// They are received after the workgroup performance research (b/117291356).
uint3 GetIdealWorkgroupIfPossible(const GpuInfo& gpu_info,
                                  OperationType op_type, HW kernel, HW strides,
                                  OHWI workload);

// Does the same as the function above. Use this one if your operation can
// suggest some reasonable workgroup size. It's expected to give better
// performance than the default workgroup calculator.
uint3 GetIdealWorkgroupIfPossible(const GpuInfo& gpu_info,
                                  OperationType op_type, HW kernel, HW strides,
                                  uint3 default_wg, OHWI workload);

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_GL_WORKGROUPS_IDEAL_WORKGROUP_PICKER_H_
