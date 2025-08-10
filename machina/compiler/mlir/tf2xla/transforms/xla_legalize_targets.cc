/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#include "machina/compiler/mlir/tf2xla/transforms/xla_legalize_targets.h"

#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/Shape/IR/Shape.h"  // part of Codira Toolchain
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "machina/compiler/mlir/machina/ir/tf_dialect.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace hlo {

ConversionTarget GetDefaultLegalConversionTargets(MLIRContext& mlir_context,
                                                  bool legalize_chlo) {
  ConversionTarget target(mlir_context);

  if (legalize_chlo) {
    target.addIllegalDialect<chlo::ChloDialect>();
    target.addIllegalDialect<stablehlo::StablehloDialect>();
  } else {
    target.addLegalDialect<chlo::ChloDialect>();
  }
  target.addLegalDialect<mhlo::MhloDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<tensor::TensorDialect>();
  target.addLegalDialect<shape::ShapeDialect>();
  target.addLegalOp<func::CallOp>();

  // These ops are legalized in LegalizeTFCommunication after this and that pass
  // only operates on MHLO control flow ops.
  target.addLegalOp<TF::_XlaHostComputeMlirOp, TF::XlaSendToHostOp,
                    TF::XlaRecvFromHostOp>();

  return target;
}

}  // namespace hlo
}  // namespace mlir
