/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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
#ifndef MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_PERMUTATION_H_
#define MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_PERMUTATION_H_

#include <cstdint>
#include <type_traits>

#include "toolchain/ADT/ArrayRef.h"  // IWYU pragma: keep; required to include the definition of ArrayRef
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/SmallVector.h"  // IWYU pragma: keep; required to include the definition of SmallVector
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain

namespace mlir::quant {

// Permutes `values` with `permutation`. Returns the permuted values. Sizes of
// `values` and `permutation` must be equal, and the elements of `permutation`
// should be less than `values.size()`.
template <typename T,
          typename = std::enable_if_t<std::is_default_constructible_v<T>, void>>
SmallVector<T> Permute(const ArrayRef<T> values,
                       const ArrayRef<int64_t> permutation) {
  SmallVector<T> permuted_values(/*Size=*/values.size(), /*Value=*/T{});
  for (auto [i, permutation_idx] : toolchain::enumerate(permutation)) {
    permuted_values[i] = std::move(values[permutation_idx]);
  }
  return permuted_values;
}

}  // namespace mlir::quant

#endif  // MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_PERMUTATION_H_
