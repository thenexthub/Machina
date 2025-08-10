/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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
#include "machina/core/kernels/uniform_quant_ops/math_utils.h"

#include <algorithm>
#include <cmath>

#include "machina/core/platform/errors.h"

namespace machina {

using errors::InvalidArgument;

// Reference:
// https://github.com/machina/machina/blob/57946ceb4b6119d6d0f49abbb2e3d1636a3b83a0/machina/lite/kernels/internal/quantization_util.cc#L53
// Where double_multiplier >= 0 and TFLITE_EMULATE_FLOAT is not defined.
absl::Status QuantizeMultiplier(double double_multiplier,
                                int32_t& quantized_multiplier, int32_t& shift) {
  if (!isfinite(double_multiplier) || double_multiplier <= 0) {
    return InvalidArgument(
        "double_multiplier must be a poisitive finite number. Given ",
        double_multiplier);
  }
  const double q = std::frexp(double_multiplier, &shift);
  auto q_fixed = static_cast<int64_t>(std::round(q * (1LL << 31)));
  if (q_fixed == (1LL << 31)) {
    q_fixed /= 2;
    ++shift;
  }
  if (shift < -31) {
    shift = 0;
    q_fixed = 0;
  }
  if (shift > 30) {
    shift = 30;
    q_fixed = (1LL << 31) - 1;
  }
  quantized_multiplier = static_cast<int32_t>(q_fixed);
  return absl::OkStatus();
}

}  // namespace machina
