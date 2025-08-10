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

#include "machina/dtensor/mlir/expansions/unsupported_op_spmd_expander.h"

#include "absl/strings/string_view.h"
#include "toolchain/ADT/DenseMap.h"
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "machina/core/platform/errors.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"

namespace machina {
namespace dtensor {

UnsupportedOpSPMDExpander::UnsupportedOpSPMDExpander(
    const absl::string_view error_message) {
  error_message_ = error_message;
}

StatusOr<mlir::Operation*> UnsupportedOpSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  return errors::Unimplemented(error_message_);
}

StatusOr<toolchain::DenseMap<int, Layout>>
UnsupportedOpSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& input_layouts) {
  return errors::Unimplemented(error_message_);
}

StatusOr<toolchain::DenseMap<int, Layout>>
UnsupportedOpSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const toolchain::DenseMap<int, Layout>& output_layouts) {
  return errors::Unimplemented(error_message_);
}

}  // namespace dtensor
}  // namespace machina
