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

#include "machina/lite/kernels/control_flow_common.h"

#include <algorithm>
#include <vector>

namespace tflite {
namespace ops {
namespace builtin {

int OutputIsInput(int output_idx, const std::vector<int>& subgraph_inputs) {
  auto e =
      std::find(subgraph_inputs.begin(), subgraph_inputs.end(), output_idx);
  return (e != subgraph_inputs.end()) ? (e - subgraph_inputs.begin()) : -1;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
