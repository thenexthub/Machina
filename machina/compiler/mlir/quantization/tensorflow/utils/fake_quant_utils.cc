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

// Copied and modified from
// //third_party/machina/compiler/mlir/lite/utils/fake_quant_utils.cc
#include "machina/compiler/mlir/quantization/machina/utils/fake_quant_utils.h"

#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops_a_m.h"

namespace mlir {
namespace quant {

// Three instances of the rule to cover the three different types of
// TF::FakeQuant operators
using PreparePerTensorFakeQuant = ConvertFakeQuantOpToQuantOps<
    TF::FakeQuantWithMinMaxVarsOp, /*PerAxis=*/false,
    FetchConstantMinMaxInputs<TF::FakeQuantWithMinMaxVarsOp>>;

using PreparePerChannelFakeQuant = ConvertFakeQuantOpToQuantOps<
    TF::FakeQuantWithMinMaxVarsPerChannelOp, /*PerAxis=*/true,
    FetchConstantMinMaxInputs<TF::FakeQuantWithMinMaxVarsPerChannelOp>>;

using PreparePerTensorFakeQuantWithMinMaxArgs = ConvertFakeQuantOpToQuantOps<
    TF::FakeQuantWithMinMaxArgsOp, /*PerAxis=*/false,
    FetchMinMaxAttrs<TF::FakeQuantWithMinMaxArgsOp>>;

// Removes the wrapper of the tf.FakeQuant* ops and creates the quant.qcast
// and quant.dcast pairs before tf.FakeQuant* ops are being foled.
LogicalResult ConvertFakeQuantOps(func::FuncOp func, MLIRContext* ctx,
                                  bool use_fake_quant_num_bits) {
  OpBuilder builder(func);

  // Insert the quant.qcast/quant.dcast ops in place of the tf.FakeQuant* ops to
  // preserve the quantization parameters.
  func.walk([&](Operation* op) {
    if (auto fake_quant = toolchain::dyn_cast<TF::FakeQuantWithMinMaxArgsOp>(op)) {
      (void)PreparePerTensorFakeQuantWithMinMaxArgs(use_fake_quant_num_bits)
          .matchAndRewrite(fake_quant, builder);
    } else if (auto fake_quant =
                   toolchain::dyn_cast<TF::FakeQuantWithMinMaxVarsOp>(op)) {
      (void)PreparePerTensorFakeQuant(use_fake_quant_num_bits)
          .matchAndRewrite(fake_quant, builder);
    } else if (auto fake_quant =
                   toolchain::dyn_cast<TF::FakeQuantWithMinMaxVarsPerChannelOp>(
                       op)) {
      (void)PreparePerChannelFakeQuant(use_fake_quant_num_bits)
          .matchAndRewrite(fake_quant, builder);
    }
  });

  return success();
}

}  // namespace quant
}  // namespace mlir
