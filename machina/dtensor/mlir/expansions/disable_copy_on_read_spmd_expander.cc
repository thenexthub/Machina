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

#include "machina/dtensor/mlir/expansions/disable_copy_on_read_spmd_expander.h"

#include "toolchain/ADT/DenseMap.h"
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/shape_utils.h"

namespace machina {
namespace dtensor {

StatusOr<mlir::Operation*> DisableCopyOnReadSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  return InferSPMDExpandedLocalShape(op);
}

StatusOr<toolchain::DenseMap<int, Layout>>
DisableCopyOnReadSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& input_layouts) {
  // DisableCopyOnRead has no outputs;
  return toolchain::DenseMap<int, Layout>();
}

StatusOr<toolchain::DenseMap<int, Layout>>
DisableCopyOnReadSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& operand_layouts,
    const toolchain::DenseMap<int, Layout>& output_layouts) {
  // Prefer the layout from operand zero.
  return operand_layouts;
}

}  // namespace dtensor
}  // namespace machina
