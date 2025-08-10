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

#ifndef MACHINA_LITE_DELEGATES_GPU_COMMON_UTIL_H_
#define MACHINA_LITE_DELEGATES_GPU_COMMON_UTIL_H_

#include "machina/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

// @param n must be non negative
// @param divisor must be greater than zero
template <typename T, typename N>
T DivideRoundUp(T n, N divisor) {
  const T div = static_cast<T>(divisor);
  const T q = n / div;
  return n % div == 0 ? q : q + 1;
}

template <>
inline uint3 DivideRoundUp(uint3 n, uint3 divisor) {
  return uint3(DivideRoundUp(n.x, divisor.x), DivideRoundUp(n.y, divisor.y),
               DivideRoundUp(n.z, divisor.z));
}

// @param number or its components must be greater than zero
// @param n must be greater than zero
template <typename T, typename N>
T AlignByN(T number, N n) {
  return DivideRoundUp(number, n) * n;
}

}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_COMMON_UTIL_H_
