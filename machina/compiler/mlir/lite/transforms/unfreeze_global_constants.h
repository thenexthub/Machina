/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_LITE_TRANSFORMS_UNFREEZE_GLOBAL_CONSTANTS_H_
#define MACHINA_COMPILER_MLIR_LITE_TRANSFORMS_UNFREEZE_GLOBAL_CONSTANTS_H_

#include "mlir/IR/DialectRegistry.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/transforms/pass.h"
#include "machina/compiler/mlir/machina/ir/tf_dialect.h"
#include "machina/compiler/mlir/machina/ir/tf_saved_model.h"

namespace mlir {
namespace TFL {

// This pass "unfreezes" the use of global constant tensor ops found in the
// module and converts them to `tf.VarHandleOp`s. Also, an initialization
// pattern `tf.AssignVariableOp(tf.VarHandleOp, tf.ConstOp)` is inserted to the
// initializer function of type "init_op" for each of the unfrozen constants.

class UnfreezeMutableGlobalTensorsPass
    : public Pass<UnfreezeMutableGlobalTensorsPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnfreezeMutableGlobalTensorsPass)

  UnfreezeMutableGlobalTensorsPass() = default;
  UnfreezeMutableGlobalTensorsPass(const UnfreezeMutableGlobalTensorsPass&) {};

  void runOnOperation() override;
  static toolchain::StringRef GetName() {
    return "UnfreezeMutableGlobalTensorsPass";
  }
  static toolchain::StringRef GetArgument() {
    return "unfreeze-mutable-global-tensors";
  }
  static toolchain::StringRef GetDescription() {
    return "Pass to unfreeze mutable global tensor ops";
  }

 private:
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect,
                    tf_saved_model::TensorFlowSavedModelDialect>();
  }
};
}  // namespace TFL
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_LITE_TRANSFORMS_UNFREEZE_GLOBAL_CONSTANTS_H_
