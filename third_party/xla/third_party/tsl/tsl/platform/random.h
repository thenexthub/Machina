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

#ifndef MACHINA_TSL_PLATFORM_RANDOM_H_
#define MACHINA_TSL_PLATFORM_RANDOM_H_

#include "machina/xla/tsl/platform/types.h"

namespace tsl {
namespace random {

// Return a 64-bit random value.  Different sequences are generated
// in different processes.
uint64 New64();

// Same as previous method, but uses a different RNG for each thread.
uint64 ThreadLocalNew64();

// Return a 64-bit random value. Uses
// std::mersenne_twister_engine::default_seed as seed value.
uint64 New64DefaultSeed();

}  // namespace random
}  // namespace tsl

#endif  // MACHINA_TSL_PLATFORM_RANDOM_H_
