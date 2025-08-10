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

#ifndef MACHINA_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_MIN_COST_FLOW_ASSIGNMENT_H_
#define MACHINA_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_MIN_COST_FLOW_ASSIGNMENT_H_

#include <stddef.h>

#include <vector>

#include "machina/lite/delegates/gpu/common/memory_management/types.h"
#include "machina/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {

// Implements memory management with a Minimum-cost flow matching algorithm.
//
// The problem of memory management is NP-complete. This function creates
// auxiliary flow graph, find minimum-cost flow in it and calculates the
// assignment of shared objects to tensors, using the result of the flow
// algorithm.
absl::Status MinCostFlowAssignment(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    ObjectsAssignment<size_t>* assignment);

}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_MIN_COST_FLOW_ASSIGNMENT_H_
