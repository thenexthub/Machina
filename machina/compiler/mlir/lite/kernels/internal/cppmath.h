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
#ifndef MACHINA_COMPILER_MLIR_LITE_KERNELS_INTERNAL_CPPMATH_H_
#define MACHINA_COMPILER_MLIR_LITE_KERNELS_INTERNAL_CPPMATH_H_

#include <cmath>

// LINT.IfChange

namespace tflite_migration {

#if defined(TF_LITE_USE_GLOBAL_CMATH_FUNCTIONS) || \
    (defined(__ANDROID__) && !defined(__NDK_MAJOR__)) || defined(__ZEPHYR__)
#define TF_LITE_GLOBAL_STD_PREFIX
#else
#define TF_LITE_GLOBAL_STD_PREFIX std
#endif

#define DECLARE_STD_GLOBAL_SWITCH1(tf_name, std_name) \
  template <class T>                                  \
  inline T tf_name(const T x) {                       \
    return TF_LITE_GLOBAL_STD_PREFIX::std_name(x);    \
  }

DECLARE_STD_GLOBAL_SWITCH1(TfLiteRound, round)

}  // namespace tflite_migration

// LINT.ThenChange(//machina/lite/kernels/internal/cppmath.h)

#endif  // MACHINA_COMPILER_MLIR_LITE_KERNELS_INTERNAL_CPPMATH_H_
