/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#ifndef MACHINA_TSL_PLATFORM_SETROUND_H_
#define MACHINA_TSL_PLATFORM_SETROUND_H_

#if defined(__ANDROID_API__) && (__ANDROID_API__ < 21)
// The <cfenv> header is broken pre-API 21 for several NDK releases.
#define TF_BROKEN_CFENV
#endif

#if defined(TF_BROKEN_CFENV)
#include <fenv.h>  // NOLINT
#else
#include <cfenv>  // NOLINT
#endif

#include "machina/xla/tsl/platform/macros.h"

namespace tsl {
namespace port {

// While this class is active, floating point rounding mode is set to the given
// mode. The mode can be one of the modes defined in <cfenv>, i.e. FE_DOWNWARD,
// FE_TONEAREST, FE_TOWARDZERO, or FE_UPWARD. The destructor restores the
// original rounding mode if it could be determined. If the original rounding
// mode could not be determined, the destructor sets it to FE_TONEAREST.
class ScopedSetRound {
 public:
  ScopedSetRound(int mode);
  ~ScopedSetRound();

 private:
  int original_mode_;

  ScopedSetRound(const ScopedSetRound&) = delete;
  void operator=(const ScopedSetRound&) = delete;
};

}  // namespace port
}  // namespace tsl

#endif  // MACHINA_TSL_PLATFORM_SETROUND_H_
