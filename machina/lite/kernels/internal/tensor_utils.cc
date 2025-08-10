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
#include "machina/lite/kernels/internal/tensor_utils.h"

#include "machina/lite/kernels/internal/optimized/neon_check.h"

#if defined(__SSSE3__) && !defined(TF_LITE_STATIC_MEMORY)
#include "machina/lite/kernels/internal/optimized/sse_tensor_utils.h"
#elif defined(USE_NEON) && !defined(TF_LITE_STATIC_MEMORY)
#include "machina/lite/kernels/internal/optimized/neon_tensor_utils.h"
#else
#include "machina/lite/kernels/internal/reference/portable_tensor_utils.h"
#endif  // __SSSE3__ or USE_NEON
