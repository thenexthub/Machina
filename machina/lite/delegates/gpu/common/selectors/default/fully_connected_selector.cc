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

#include <memory>
#include <utility>

#include "machina/lite/delegates/gpu/common/gpu_info.h"
#include "machina/lite/delegates/gpu/common/operations.h"
#include "machina/lite/delegates/gpu/common/shape.h"
#include "machina/lite/delegates/gpu/common/task/gpu_operation.h"
#include "machina/lite/delegates/gpu/common/tasks/conv_generic.h"
#include "machina/lite/delegates/gpu/common/tasks/fully_connected.h"

namespace tflite {
namespace gpu {

std::unique_ptr<GPUOperation> SelectFullyConnected(
    const FullyConnectedAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def, int batch_size) {
  if (op_def.IsBatchSupported()) {
    BHWC dst_shape = BHWC(batch_size, 1, 1, attr.weights.shape.o);
    ConvGeneric conv = CreateConvGeneric(gpu_info, op_def, attr, &dst_shape);
    return std::make_unique<ConvGeneric>(std::move(conv));
  } else {
    FullyConnected fc = CreateFullyConnected(gpu_info, op_def, attr);
    return std::make_unique<FullyConnected>(std::move(fc));
  }
}

std::unique_ptr<GPUOperation> SelectFullyConnected(
    const FullyConnectedInt8Attributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def) {
  FullyConnected fc = CreateFullyConnected(gpu_info, op_def, attr);
  return std::make_unique<FullyConnected>(std::move(fc));
}

}  // namespace gpu
}  // namespace tflite
