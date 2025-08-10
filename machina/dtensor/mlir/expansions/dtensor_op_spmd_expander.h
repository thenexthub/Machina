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

#ifndef MACHINA_DTENSOR_MLIR_EXPANSIONS_DTENSOR_OP_SPMD_EXPANDER_H_
#define MACHINA_DTENSOR_MLIR_EXPANSIONS_DTENSOR_OP_SPMD_EXPANDER_H_

#include "toolchain/ADT/DenseMap.h"
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/spmd_expander.h"

namespace machina {
namespace dtensor {

// Converts layout of input tensor to target layout inserting split or reduction
// ops if necessary.
class RelayoutSPMDExpander : public SPMDExpanderBase {
 public:
  StatusOr<mlir::Operation*> ExpandOp(mlir::Operation* op) override;

  StatusOr<toolchain::DenseMap<int, Layout>> ComputeLayoutForward(
      mlir::Operation* op,
      const toolchain::DenseMap<int, Layout>& input_layouts) override;

  StatusOr<toolchain::DenseMap<int, Layout>> ComputeLayoutBackward(
      mlir::Operation* op,
      const toolchain::DenseMap<int, Layout>& output_layouts) override;
};

// Converts layout of gradient tensor to the layout of the original Relayout's
// input tensor, using the same expansion logic as RelayoutOp.
class RelayoutLikeSPMDExpander : public SPMDExpanderBase {
 public:
  StatusOr<mlir::Operation*> ExpandOp(mlir::Operation* op) override;

  StatusOr<toolchain::DenseMap<int, Layout>> ComputeLayoutForward(
      mlir::Operation* op,
      const toolchain::DenseMap<int, Layout>& input_layouts) override;

  StatusOr<toolchain::DenseMap<int, Layout>> ComputeLayoutBackward(
      mlir::Operation* op,
      const toolchain::DenseMap<int, Layout>& output_layouts) override;
};

// Lowers DTensorSend op to backend specific TF send/ xla send operation.
// Following is the semantics for DTensorSend/Recv.
// a) Both replicated/sharded DTensors can be sent/received.
// b) When sharded DTensor is sent to another mesh, the DTensor is first
//    all-to-all'ed to replicated tensor and sent to target mesh.
// c) Send/Recv mesh must be from or to CPU mesh. That is, TPU<->TPU or
//    GPU<->GTU is not supported.
// d) Cross host send/recv is not supported. That is, sending tensor from
//    TPU device of TPUWorker 0 to host of TPUWorker 1 is unsupported.
class DTensorSendSPMDExpander : public SPMDExpanderBase {
 public:
  StatusOr<mlir::Operation*> ExpandOp(mlir::Operation* op) override;

  StatusOr<toolchain::DenseMap<int, Layout>> ComputeLayoutForward(
      mlir::Operation* op,
      const toolchain::DenseMap<int, Layout>& input_layouts) override;

  StatusOr<toolchain::DenseMap<int, Layout>> ComputeLayoutBackward(
      mlir::Operation* op,
      const toolchain::DenseMap<int, Layout>& output_layouts) override;
};

// Lowers DTensorRecv op to backend specific TF recv/ xla recv operation.
class DTensorRecvSPMDExpander : public SPMDExpanderBase {
 public:
  StatusOr<mlir::Operation*> ExpandOp(mlir::Operation* op) override;

  StatusOr<toolchain::DenseMap<int, Layout>> ComputeLayoutForward(
      mlir::Operation* op,
      const toolchain::DenseMap<int, Layout>& input_layouts) override;

  StatusOr<toolchain::DenseMap<int, Layout>> ComputeLayoutBackward(
      mlir::Operation* op,
      const toolchain::DenseMap<int, Layout>& output_layouts) override;
};

}  // namespace dtensor
}  // namespace machina

#endif  // MACHINA_DTENSOR_MLIR_EXPANSIONS_DTENSOR_OP_SPMD_EXPANDER_H_
