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
#ifndef MACHINA_LITE_KERNELS_INTERNAL_REFERENCE_FLOOR_MOD_H_
#define MACHINA_LITE_KERNELS_INTERNAL_REFERENCE_FLOOR_MOD_H_

#include <cmath>
#include <functional>

namespace tflite {

namespace reference_ops {

template <typename T>
T FloorMod(T input1, T input2) {
  struct FloatMod {
    float operator()(const float lhs, const float rhs) const {
      return std::fmod(lhs, rhs);
    }
  };
  using ModFunc = typename std::conditional<std::is_integral<T>::value,
                                            std::modulus<T>, FloatMod>::type;
  ModFunc mod_func;
  T trunc_mod = mod_func(input1, input2);
  return (trunc_mod != 0) && ((input2 < 0) != (trunc_mod < 0))
             ? (trunc_mod + input2)
             : trunc_mod;
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // MACHINA_LITE_KERNELS_INTERNAL_REFERENCE_FLOOR_MOD_H_
