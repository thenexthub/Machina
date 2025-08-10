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
#include <string>
namespace toco {

bool StrContains(const std::string& x, const std::string& search_pattern) {
  return x.find(search_pattern) != std::string::npos;
}

void Transpose2DTensor(const float* tensor, int row, int col,
                       float* transposed_tensor) {
  float* result = transposed_tensor;
  for (int r = 0; r < row; ++r) {
    for (int c = 0; c < col; ++c) {
      *(result + c * row) = *tensor++;
    }
    ++result;
  }
}

}  // end namespace toco
