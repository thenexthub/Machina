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

#ifndef MACHINA_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_KISS_FFT_COMMON_H_
#define MACHINA_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_KISS_FFT_COMMON_H_

// This header file should be included in all variants of kiss_fft_$type.{h,cc}
// so that their sub-included source files do not mistakenly wrap libc header
// files within their kissfft_$type namespaces.
// E.g, This header avoids kissfft_int16.h containing:
//   namespace kiss_fft_int16 {
//     #include "kiss_fft.h"
//   }
// where kiss_fft_.h contains:
//   #include <math.h>
//
// TRICK: By including the following header files here, their preprocessor
// header guards prevent them being re-defined inside of the kiss_fft_$type
// namespaces declared within the kiss_fft_$type.{h,cc} sources.
// Note that the original kiss_fft*.h files are untouched since they
// may be used in libraries that include them directly.

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef FIXED_POINT
#include <sys/types.h>
#endif

#ifdef USE_SIMD
#include <xmmintrin.h>
#endif
#endif  // MACHINA_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_KISS_FFT_COMMON_H_
