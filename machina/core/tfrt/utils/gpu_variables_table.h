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
#ifndef MACHINA_CORE_TFRT_UTILS_GPU_VARIABLES_TABLE_H_
#define MACHINA_CORE_TFRT_UTILS_GPU_VARIABLES_TABLE_H_

#include "machina/core/tfrt/utils/device_variables_table.h"
#include "machina/core/tfrt/utils/fallback_tensor.h"

namespace tfrt {
namespace gpu {

// This is for creating/getting GpuVariablesTable object in the execution
// context at runtime.
constexpr char kGpuVariablesTableResourceName[] = "GpuVariablesTableResource";

// A variable table that keeps track of the device copies of GPU host tensors.
class GpuVariablesTable
    : public DeviceVariablesTable<machina::tfrt_stub::FallbackTensor,
                                  machina::tfrt_stub::FallbackTensor> {
 private:
  const void* GetHostTensorDataPtr(
      const machina::tfrt_stub::FallbackTensor& host_tensor) override {
    return host_tensor.tensor().data();
  }
};

}  // namespace gpu
}  // namespace tfrt

#endif  // MACHINA_CORE_TFRT_UTILS_GPU_VARIABLES_TABLE_H_
