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

#ifndef MACHINA_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_KISS_FFT_INT16_H_
#define MACHINA_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_KISS_FFT_INT16_H_

#include "machina/lite/experimental/microfrontend/lib/kiss_fft_common.h"

// Wrap 16-bit kiss fft in its own namespace. Enables us to link an application
// with different kiss fft resultions (16/32 bit interger, float, double)
// without getting a linker error.
#define FIXED_POINT 16
namespace kissfft_fixed16 {
#include "kiss_fft.h"
#include "tools/kiss_fftr.h"
}  // namespace kissfft_fixed16
#undef FIXED_POINT
#undef kiss_fft_scalar
#undef KISS_FFT_H

#endif  // MACHINA_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_KISS_FFT_INT16_H_
