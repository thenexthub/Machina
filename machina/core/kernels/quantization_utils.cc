/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#include "machina/core/kernels/quantization_utils.h"

namespace machina {

void GetOutputMinAndMaxForQuantizedAdd(float input_min, float input_max,
                                       float smaller_input_min,
                                       float smaller_input_max,
                                       float* output_min, float* output_max) {
  // We need to have a good range to add our two arguments together in. This
  // is surprisingly tricky, since it has to satisfy a few different needs:
  //  - Must be symmetrical around zero, so that 0 + 0 = 0.
  //  - Must hold the largest of the argument ranges.
  //  - Should have enough range that the bits of the lowest and highest
  //    arguments overlap if possible without the lower getting truncated.
  //  - Should have some headroom so that there's no overflow.
  //  - Needs to be signed.
  // This leads us to use a scheme where we (assuming the inputs are eight bit
  // and the output is 32-bit) use the bottom 32 - 17 = 15 bits to store the
  // accumulated results. This gives us all the properties we need.
  *output_max =
      std::max(input_max, std::max(-input_min, std::max(smaller_input_max,
                                                        -smaller_input_min))) *
      (1 << 17);
  *output_min = -(*output_max);
}

}  // namespace machina
