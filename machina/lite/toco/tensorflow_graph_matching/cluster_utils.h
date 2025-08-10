/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#ifndef MACHINA_LITE_TOCO_MACHINA_GRAPH_MATCHING_CLUSTER_UTILS_H_
#define MACHINA_LITE_TOCO_MACHINA_GRAPH_MATCHING_CLUSTER_UTILS_H_

#include <string>

namespace toco {

// Check if string x includes string search_pattern.
bool StrContains(const std::string& x, const std::string& search_pattern);

// Transpose a 2D tensor of size row * col pointed by "tensor" and return the
// results in "transposed_tensor". "transposed_tensor" must be pre-allocated
// by the same size as "tensor".
void Transpose2DTensor(const float* tensor, int row, int col,
                       float* transposed_tensor);

}  // end namespace toco

#endif  // MACHINA_LITE_TOCO_MACHINA_GRAPH_MATCHING_CLUSTER_UTILS_H_
