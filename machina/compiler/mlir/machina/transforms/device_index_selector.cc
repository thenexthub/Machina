/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

// Converts DeviceIndex to constant device.

#include <memory>

#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/transforms/passes.h"

namespace mlir {
namespace TF {
namespace {

#define GEN_PASS_DEF_DEVICEINDEXSELECTORPASS
#include "machina/compiler/mlir/machina/transforms/tf_passes.h.inc"

// Folds the DeviceIndex op to a constant value. The DeviceIndex return the
// index of the device the op should run on. The user can use this to provide
// different op specializations. E.g.,
//
// ```mlir
//  %1 = "tf.DeviceIndex"()
//          {device = "", device_names = ["CPU", "GPU"]} : () -> tensor<i32>
//  %4 = "tf.Case"(%1, %arg0, %arg1)
//          {branches = [@foo, @baz], output_shapes = [#tf_type.shape<>]} :
//            (tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<f32>
// ```
//
// Shows an example where there are 2 different functions which could be
// executed to produce the same values but with different functions optimized
// for CPU or GPU.
struct DeviceIndexSelector
    : public impl::DeviceIndexSelectorPassBase<DeviceIndexSelector> {
  void runOnOperation() override;
};

}  // namespace

void DeviceIndexSelector::runOnOperation() {
  func::FuncOp func = getOperation();
  // Convert all the DeviceIndex ops to constant values.
  func.getBody().walk([](TF::DeviceIndexOp op) {
    // This just selects the default in all cases where DeviceIndex feeds into
    // tf.Case. This could be enhanced to have some sort of policy in the
    // future.
    OpBuilder b(op);
    RankedTensorType type = RankedTensorType::get({}, b.getIntegerType(32));
    int index = op.getDeviceNames().size();
    for (auto use : op.getOperation()->getUsers()) {
      // Skip if it doesn't feed into case. Alternatively this could always
      // return the CPU device index if it exists.
      if (!isa<TF::CaseOp>(use)) return;
    }
    DenseElementsAttr attr =
        DenseElementsAttr::get(type, b.getI32IntegerAttr(index));
    auto constant = arith::ConstantOp::create(b, op.getLoc(), type, attr);
    op.replaceAllUsesWith(constant.getOperation());
    op.erase();
  });
}

// Creates an instance of the TensorFlow DeviceIndex selector pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateDeviceIndexSelectorPass() {
  return std::make_unique<DeviceIndexSelector>();
}

}  // namespace TF
}  // namespace mlir
