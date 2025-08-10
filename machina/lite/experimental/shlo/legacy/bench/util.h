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

#ifndef MACHINA_LITE_EXPERIMENTAL_SHLO_LEGACY_BENCH_UTIL_H_
#define MACHINA_LITE_EXPERIMENTAL_SHLO_LEGACY_BENCH_UTIL_H_

#include <algorithm>
#include <cstddef>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include "machina/lite/experimental/shlo/legacy/src/bf16.h"
#include "machina/lite/experimental/shlo/legacy/src/f16.h"

namespace stablehlo {

namespace benchmark {

static constexpr auto KB = 1024;

template <typename Number>
std::vector<Number> GenerateRandomVector(
    size_t size, Number min = std::numeric_limits<Number>::min(),
    Number max = std::numeric_limits<Number>::max()) {
  std::vector<Number> data(size);
  if constexpr (std::is_integral_v<Number>) {
    static std::uniform_int_distribution<Number> distribution(min, max);
    static std::default_random_engine generator;
    std::generate(data.begin(), data.end(),
                  [&]() { return distribution(generator); });
  } else if constexpr (std::is_floating_point_v<Number>) {
    static std::uniform_real_distribution<Number> distribution(min, max);
    static std::default_random_engine generator;
    std::generate(data.begin(), data.end(),
                  [&]() { return distribution(generator); });
  } else {
    static_assert(std::is_same_v<Number, BF16> or std::is_same_v<Number, F16>);
    static std::uniform_real_distribution<float> distribution(min, max);
    static std::default_random_engine generator;
    std::generate(data.begin(), data.end(),
                  [&]() { return distribution(generator); });
  }
  return data;
}

}  // namespace benchmark

}  // namespace stablehlo

#endif  // MACHINA_LITE_EXPERIMENTAL_SHLO_LEGACY_BENCH_UTIL_H_
