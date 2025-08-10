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
#include "machina/compiler/mlir/lite/quantization/lite/tfl_to_std.h"

#include "toolchain/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"
#include "machina/compiler/mlir/lite/quantization/common/quantization_lib/quantization_utils.h"
#include "machina/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "machina/compiler/mlir/lite/utils/utils.h"

namespace mlir {
namespace TFL {

void ConvertTFLQuantOpsToMlirQuantOps(func::FuncOp func) {
  OpBuilder b(func);
  func.walk([&](Operation* op) {
    b.setInsertionPoint(op);
    if (auto dq = toolchain::dyn_cast<DequantizeOp>(op)) {
      auto dcast = b.create<quantfork::DequantizeCastOp>(
          dq.getLoc(), dq.getOutput().getType(), dq.getInput());
      dq.getOutput().replaceAllUsesWith(dcast);
      dq.erase();
    } else if (auto q = toolchain::dyn_cast<QuantizeOp>(op)) {
      auto qcast = b.create<quantfork::QuantizeCastOp>(
          q.getLoc(), q.getOutput().getType(), q.getInput());
      q.getOutput().replaceAllUsesWith(qcast);
      q.erase();
    } else if (auto q = toolchain::dyn_cast<ConstOp>(op)) {
      auto value = q.getValue();
      auto type = q.getResult().getType();
      if (arith::ConstantOp::isBuildableWith(value, type)) {
        auto c = b.create<arith::ConstantOp>(q.getLoc(), q.getValue());
        q.getOutput().replaceAllUsesWith(c);
        q.erase();
      } else if (TFL::NoValueOp::isBuildableWith(value, type)) {
        auto c = b.create<TFL::NoValueOp>(q.getLoc(), type, mlir::UnitAttr());
        q.getOutput().replaceAllUsesWith(c);
        q.erase();
      }
    }
  });
}

void ConvertMlirQuantOpsToTFLQuantOps(func::FuncOp func) {
  OpBuilder b(func);
  func.walk([&](Operation* op) {
    b.setInsertionPoint(op);
    if (auto dq = toolchain::dyn_cast<quantfork::DequantizeCastOp>(op)) {
      auto dcast = b.create<DequantizeOp>(dq.getLoc(), dq.getResult().getType(),
                                          dq.getArg());
      dq.getResult().replaceAllUsesWith(dcast);
      if (auto extra_attr = op->getAttr(kVolatileOpAttrName)) {
        dcast->setAttr(kVolatileOpAttrName, extra_attr);
      }
      dq.erase();
    } else if (auto q = toolchain::dyn_cast<quantfork::QuantizeCastOp>(op)) {
      auto out_type = q.getResult().getType();
      auto qcast = b.create<QuantizeOp>(q.getLoc(), out_type, q.getArg(),
                                        TypeAttr::get(out_type));
      q.getResult().replaceAllUsesWith(qcast);
      if (auto extra_attr = op->getAttr(kVolatileOpAttrName)) {
        qcast->setAttr(kVolatileOpAttrName, extra_attr);
      }
      q.erase();
    }
  });
}

}  // namespace TFL
}  // namespace mlir
