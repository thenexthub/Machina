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

#ifndef MACHINA_DTENSOR_MLIR_EXPANSIONS_DATAPARALLEL_SPMD_EXPANDER_H_
#define MACHINA_DTENSOR_MLIR_EXPANSIONS_DATAPARALLEL_SPMD_EXPANDER_H_
#include <utility>
#include <vector>

#include "toolchain/ADT/DenseMap.h"
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/spmd_expander.h"

namespace machina {
namespace dtensor {

// General SPMD Expander for data parallel ops.

// We define data parallel ops as ops that have tensors possibly with a batch
// dimension. Assumes batch dimensions start from the left. Tensors may
// may have multiple batch dimensions, including zero
class DataparallelSPMDExpander : public SPMDExpanderBase {
 protected:
  // These maps contain {arg_index, non_batch_rank}
  // Example is for TF:FFT2D, the batchable_operands and batchable_outputs has
  // {0, 2} because the first argument is batchable and the last 2 dimensions
  // are the non-batch dimensions
  toolchain::DenseMap<int, int> batchable_operands_;
  toolchain::DenseMap<int, int> batchable_outputs_;
  StatusOr<mlir::Operation*> ExpandOp(mlir::Operation* op) override;

  StatusOr<toolchain::DenseMap<int, Layout>> ComputeLayoutForward(
      mlir::Operation* op,
      const toolchain::DenseMap<int, Layout>& input_layouts) override;

  StatusOr<toolchain::DenseMap<int, Layout>> ComputeLayoutBackward(
      mlir::Operation* op,
      const toolchain::DenseMap<int, Layout>& output_layouts) override;

 public:
  explicit DataparallelSPMDExpander(toolchain::DenseMap<int, int> batchable_operands,
                                    toolchain::DenseMap<int, int> batchable_outputs)
      : batchable_operands_(std::move(batchable_operands)),
        batchable_outputs_(std::move(batchable_outputs)) {}

 private:
  // Relayouts all operands and outputs with a batch dimensions to a batch
  // sharded layout. This should only be called when there is at least one
  // batch sharded operand or batch sharded output
  StatusOr<mlir::Operation*> RelayoutOperandsAndOutputs(
      mlir::Operation* op, const std::vector<Layout>& operand_layouts,
      const std::vector<Layout>& output_layouts);
};
}  // namespace dtensor
}  // namespace machina

#endif  // MACHINA_DTENSOR_MLIR_EXPANSIONS_DATAPARALLEL_SPMD_EXPANDER_H_
