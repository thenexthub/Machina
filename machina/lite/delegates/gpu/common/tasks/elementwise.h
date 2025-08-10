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

#ifndef MACHINA_LITE_DELEGATES_GPU_COMMON_TASKS_ELEMENTWISE_H_
#define MACHINA_LITE_DELEGATES_GPU_COMMON_TASKS_ELEMENTWISE_H_

#include <string>

#include "machina/lite/delegates/gpu/common/data_type.h"
#include "machina/lite/delegates/gpu/common/gpu_info.h"
#include "machina/lite/delegates/gpu/common/operations.h"
#include "machina/lite/delegates/gpu/common/precision.h"
#include "machina/lite/delegates/gpu/common/shape.h"
#include "machina/lite/delegates/gpu/common/status.h"
#include "machina/lite/delegates/gpu/common/task/gpu_operation.h"

namespace tflite {
namespace gpu {

// Creates simple one input operation without any parameters, for example
// log, sin, cos, etc.
ElementwiseDescriptor CreateElementwiseOneInput(const GpuInfo& gpu_info,
                                                CalculationsPrecision precision,
                                                const OperationType& op_type);

GPUOperation CreateElementwiseOneInput(const GpuInfo& gpu_info,
                                       const OperationDef& definition,
                                       const OperationType& op_type);

// Creates simple one input operation without any parameters, for example
// log, sin, cos, etc.
// Can broadcast input.
GPUOperation CreateElementwiseOneInputWithBroadcast(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type, const BHWC& input_shape,
    const BHWC& output_shape);

// Creates simple two input(first input is runtime tensor and second input is
// constant or linear/hwc tensor) operation, for example sub, div and etc.
template <DataType DataTypeT, typename T>
GPUOperation CreateElementwise(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type,
    const ElementwiseAttributesBase<DataTypeT, T>& attr);

// Creates simple two input(first input is runtime tensor and second input is
// constant or linear/hwc tensor) operation, for example sub, div and etc.
// Can broadcast input.
template <DataType DataTypeT, typename T>
GPUOperation CreateElementwiseWithBroadcast(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type,
    const ElementwiseAttributesBase<DataTypeT, T>& attr,
    const BHWC& input_shape, const BHWC& output_shape);

// Creates simple two input(2 runtime tensors) operation, for example
// sub, div and etc.
GPUOperation CreateElementwiseTwoInput(const OperationDef& definition,
                                       const OperationType& op_type,
                                       const BHWC& shape);

// Creates simple two input(2 runtime tensors) operation, for example
// sub, div and etc.
// Can broadcast first and second input simultaneously.
GPUOperation CreateElementwiseTwoInputWithBroadcast(
    const OperationDef& definition, const OperationType& op_type,
    const BHWC& first_input_shape, const BHWC& second_input_shape,
    const BHWC& output_shape);

}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_COMMON_TASKS_ELEMENTWISE_H_
