/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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
#ifndef MACHINA_LITE_KERNELS_OP_MACROS_H_
#define MACHINA_LITE_KERNELS_OP_MACROS_H_

#include "machina/lite/micro/micro_log.h"

#if !defined(TF_LITE_MCU_DEBUG_LOG)
#include <cstdlib>
#define TFLITE_ABORT abort()
#else
inline void AbortImpl() {
  MicroPrintf("HALTED");
  while (1) {
  }
}
#define TFLITE_ABORT AbortImpl();
#endif

#if defined(NDEBUG)
#define TFLITE_ASSERT_FALSE (static_cast<void>(0))
#else
#define TFLITE_ASSERT_FALSE TFLITE_ABORT
#endif

#define TF_LITE_FATAL(msg)    \
  do {                        \
    MicroPrintf("%s", (msg)); \
    TFLITE_ABORT;             \
  } while (0)

#define TF_LITE_ASSERT(x)        \
  do {                           \
    if (!(x)) TF_LITE_FATAL(#x); \
  } while (0)

#endif  // MACHINA_LITE_KERNELS_OP_MACROS_H_
