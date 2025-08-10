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
#ifndef MACHINA_LITE_KERNELS_RNG_UTIL_H_
#define MACHINA_LITE_KERNELS_RNG_UTIL_H_

#include <array>
#include <cstdint>

namespace tflite {
namespace rng {

// Implements the ThreeFry counter-based PRNG algorithm. Use 20 rounds.
// Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
// http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
std::array<uint32_t, 2> Threefry2x32(uint32_t key_0, uint32_t key_1,
                                     std::array<uint32_t, 2> ctr);

// Implements the Philox4x32 counter-based PRNG algorithm. Use 10 rounds.
// Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
// http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
std::array<uint32_t, 4> Philox4x32(uint32_t key_0, uint32_t key_1,
                                   std::array<uint32_t, 4> ctr);

}  // namespace rng
}  // namespace tflite

#endif  // MACHINA_LITE_KERNELS_RNG_UTIL_H_
