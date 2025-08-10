/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

/* NNAPI Hardware Implementation */
#ifndef MACHINA_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_HARDWARES_NNAPI_HARDWARE_H_
#define MACHINA_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_HARDWARES_NNAPI_HARDWARE_H_

#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/experimental/tac/hardwares/simple_hardware.h"

namespace mlir {
namespace TFL {
namespace tac {

class NNAPIHardware : public SimpleHardware {
 public:
  static constexpr char kId[] = "NNAPI";

  mlir::RewritePatternSet GetTransformations(
      MLIRContext* context) const override;

  mlir::TypeID GetTypeId() const override {
    return mlir::TypeID::get<NNAPIHardware>();
  }

  bool IsNotSupportedOp(mlir::Operation* op) const override { return false; }

  float AdvantageOverCPU() const override { return 5.0; }
};

}  // namespace tac
}  // namespace TFL
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_HARDWARES_NNAPI_HARDWARE_H_
