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

// Declarations for 1D FFT routines in third_party/fft2d/fft2d.

#ifndef FFT2D_FFT_H__
#define FFT2D_FFT_H__

#ifdef __cplusplus
extern "C" {
#endif

extern void cdft(int, int, double *, int *, double *);
extern void rdft(int, int, double *, int *, double *);
extern void ddct(int, int, double *, int *, double *);
extern void ddst(int, int, double *, int *, double *);
extern void dfct(int, double *, double *, int *, double *);
extern void dfst(int, double *, double *, int *, double *);

#ifdef __cplusplus
}
#endif

#endif  // FFT2D_FFT_H__
