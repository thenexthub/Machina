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
#ifndef MACHINA_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_REDUCE_WINDOW_UTIL_H_
#define MACHINA_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_REDUCE_WINDOW_UTIL_H_

#include <cstdint>
#include <optional>

#include "toolchain/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/op_util_common.h"
#include "machina/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

// Helpers for working with mhlo.reduce_window attrs in the mlir api as
// native cc types.

namespace mlir::odml {

class ReduceWindowView {
 public:
  explicit ReduceWindowView(mhlo::ReduceWindowOp op);

  toolchain::ArrayRef<int64_t> WindowDims() const { return window_dims_; }
  int64_t WindowSize() const { return window_size_; }
  toolchain::ArrayRef<int64_t> WindowStrides() const { return window_strides_; }
  toolchain::ArrayRef<DimPadding> Paddings() const { return paddings_; }
  toolchain::ArrayRef<int64_t> WindowDilations() const { return window_dilations_; }
  toolchain::ArrayRef<int64_t> BaseDilations() const { return base_dilations_; }
  int64_t Rank() const { return rank_; }

  std::optional<Layout> GuessLayout() const;

 private:
  int64_t rank_;

  toolchain::SmallVector<int64_t, 4> window_dims_;
  toolchain::SmallVector<int64_t, 4> window_strides_;
  toolchain::SmallVector<int64_t, 4> window_dilations_;

  toolchain::SmallVector<DimPadding, 4> paddings_;

  toolchain::SmallVector<int64_t, 4> base_dilations_;

  int64_t window_size_;
};

}  // namespace mlir::odml

#endif  // MACHINA_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_REDUCE_WINDOW_UTIL_H_
