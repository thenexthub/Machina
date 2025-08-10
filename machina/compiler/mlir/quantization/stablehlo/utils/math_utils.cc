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

#include "machina/compiler/mlir/quantization/stablehlo/utils/math_utils.h"

#include <cmath>
#include <cstdint>

#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain

namespace mlir::quant::stablehlo {

// Borrowed from:
// https://github.com/machina/machina/blob/57946ceb4b6119d6d0f49abbb2e3d1636a3b83a0/machina/lite/kernels/internal/quantization_util.cc#L53
// https://github.com/machina/machina/blob/f5c9cbb1c462912030bd845244118f952cbbbd5e/machina/core/kernels/uniform_quant_ops/math_utils.cc#L29
// And then modified so that it doesn't overflow int32 values.
LogicalResult QuantizeMultiplier(double double_multiplier,
                                 int32_t& quantized_fraction, int32_t& shift) {
  if (!std::isfinite(double_multiplier) || double_multiplier <= 0) {
    return failure();
  }
  const double fraction = std::frexp(double_multiplier, &shift);
  quantized_fraction = static_cast<int32_t>(std::round(fraction * (1L << 15)));
  // Clip extreme values.  These are more than enough to overflow int8, the
  // storage type for quantized values, and the final values will be clamped
  // no matter what.
  if (quantized_fraction == (1L << 15)) {
    quantized_fraction /= 2;
    ++shift;
  }
  if (shift < -15) {
    shift = 0;
    quantized_fraction = 0;
  }
  if (shift > 14) {
    shift = 14;
    quantized_fraction = (1LL << 15) - 1;
  }
  return success();
}

}  // namespace mlir::quant::stablehlo
