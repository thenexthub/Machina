/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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

#ifndef MACHINA_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_F16_H_
#define MACHINA_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_F16_H_

#include "machina/lite/experimental/shlo/legacy/src/has_keyword.h"

#if defined(__STDCPP_FLOAT16_T__)
#include <stdfloat>
namespace stablehlo {
using F16 = float16_t;
}  // namespace stablehlo

#elif __has_keyword(_Float16)
namespace stablehlo {
using F16 = _Float16;
}  // namespace stablehlo

#else
#error Type F16 is not available
#endif

#endif  // MACHINA_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_F16_H_
