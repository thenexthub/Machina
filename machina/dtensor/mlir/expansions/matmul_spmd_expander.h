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

#ifndef MACHINA_DTENSOR_MLIR_EXPANSIONS_MATMUL_SPMD_EXPANDER_H_
#define MACHINA_DTENSOR_MLIR_EXPANSIONS_MATMUL_SPMD_EXPANDER_H_

#include <optional>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "toolchain/ADT/DenseMap.h"
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "machina/core/platform/status.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/spmd_expander.h"

namespace machina {
namespace dtensor {

class MatMulSPMDExpander : public SPMDExpanderBase {
 public:
  StatusOr<mlir::Operation*> ExpandOp(mlir::Operation* op) override;

  StatusOr<toolchain::DenseMap<int, Layout>> ComputeLayoutForward(
      mlir::Operation* op,
      const toolchain::DenseMap<int, Layout>& input_layouts) override;

  StatusOr<toolchain::DenseMap<int, Layout>> ComputeLayoutBackward(
      mlir::Operation* op,
      const toolchain::DenseMap<int, Layout>& output_layouts) override;

 private:
  StatusOr<Layout> OutputLayoutAndReducedDims(
      bool allow_unknown_layouts, mlir::Operation* op,
      absl::flat_hash_set<std::string>* reduced_dims,
      std::optional<Layout>* left, std::optional<Layout>* right);

  // This function prepares the inputs (x, y or a, b) to (Batch)MatMul by
  // possibly computing a new layout for each input that allows us to simply
  // emit a local (Batch)MatMul op. Once the layouts are computed, the function
  // calls EmitRelayout to transform from left_layout, right_layout to the
  // newly computed layouts.
  // The left and right arguments are set to the mlir::Values representing the
  // tensors with the possibly new layout.
  // reduced_dim will contain the dim that must be reduced on after the local
  // MatMul. It may be set to Layout::kUnsharded if no reduction is needed.
  // matmul_layout will be set to the layout of the output of the local matmul
  // (after the above reduction). This may be different from the desired output
  // layout.
  absl::Status MaybeRelayoutInputs(
      mlir::Operation* op, const Layout& left_layout, bool left_transposed,
      const Layout& right_layout, bool right_transposed,
      const Layout& output_layout, std::string& reduced_dim,
      Layout& matmul_layout, mlir::Value& left, mlir::Value& right);
};

}  // namespace dtensor
}  // namespace machina

#endif  // MACHINA_DTENSOR_MLIR_EXPANSIONS_MATMUL_SPMD_EXPANDER_H_
