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

#ifndef SIGNAL_SRC_KISS_FFT_WRAPPERS_KISS_FFT_FLOAT_H_
#define SIGNAL_SRC_KISS_FFT_WRAPPERS_KISS_FFT_FLOAT_H_

#include "signal/src/kiss_fft_wrappers/kiss_fft_common.h"

// Wrap floating point kiss fft in its own namespace. Enables us to link an
// application with different kiss fft resolutions
// (16/32 bit integer, float, double) without getting a linker error.
#undef FIXED_POINT
namespace kiss_fft_float {
#include "kiss_fft.h"
#include "tools/kiss_fftr.h"
}  // namespace kiss_fft_float
#undef FIXED_POINT

#endif  // SIGNAL_SRC_KISS_FFT_WRAPPERS_KISS_FFT_FLOAT_H_
