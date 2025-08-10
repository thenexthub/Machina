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
#include "machina/compiler/mlir/quantization/machina/ops/uniform_op_quant_spec.h"

#include <memory>

#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

namespace mlir::quant {

std::unique_ptr<OpQuantSpec> GetUniformOpQuantSpec(Operation* op) {
  auto spec = std::make_unique<OpQuantSpec>();
  if (isa<TF::UniformQuantizedConvolutionHybridOp>(op) ||
      isa<TF::UniformQuantizedConvolutionOp>(op)) {
    spec->coeff_op_quant_dim[1] = 3;
  } else if (isa<TF::UniformQuantizedDotHybridOp>(op)) {
    spec->coeff_op_quant_dim[1] = -1;
  }

  for (auto quantizable_operand : spec->coeff_op_quant_dim) {
    spec->quantizable_operands.insert(quantizable_operand.first);
  }
  return spec;
}

}  // namespace mlir::quant
