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

#ifndef MACHINA_LITE_DELEGATES_GPU_COMMON_SELECTORS_DEFAULT_SELECTOR_H_
#define MACHINA_LITE_DELEGATES_GPU_COMMON_SELECTORS_DEFAULT_SELECTOR_H_

#include <memory>

#include "machina/lite/delegates/gpu/common/model.h"
#include "machina/lite/delegates/gpu/common/model_hints.h"
#include "machina/lite/delegates/gpu/common/selectors/subgraph.h"
#include "machina/lite/delegates/gpu/common/status.h"
#include "machina/lite/delegates/gpu/common/task/gpu_operation.h"
#include "machina/lite/delegates/gpu/common/task/tensor_desc.h"

namespace tflite {
namespace gpu {

absl::Status SelectDefault(const GpuInfo& gpu_info, const OperationDef& op_def,
                           ModelHints hints, const std::vector<Value*>& inputs,
                           const std::vector<Value*>& outputs, const Node& node,
                           GPUOperationsSubgraph* gpu_subgraph);

}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_COMMON_SELECTORS_DEFAULT_SELECTOR_H_
