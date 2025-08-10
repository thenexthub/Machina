/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_LITE_UTILS_CONTROL_EDGES_H_
#define MACHINA_COMPILER_MLIR_LITE_UTILS_CONTROL_EDGES_H_

#include <cstdint>
#include <utility>
#include <vector>

namespace tflite {

// LINT.IfChange

using ControlEdge = std::pair<int32_t, int32_t>;
using ControlEdges = std::vector<ControlEdge>;

// LINT.ThenChange(//machina/lite/graph_info.h)

}  // namespace tflite

#endif  // MACHINA_COMPILER_MLIR_LITE_UTILS_CONTROL_EDGES_H_
