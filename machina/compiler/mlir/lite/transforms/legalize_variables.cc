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

#include <optional>
#include <utility>

#include "toolchain/Support/Casting.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/ir/tfl_ops.h"
#include "machina/compiler/mlir/lite/transforms/passes.h"
#include "machina/compiler/mlir/lite/utils/variables_utils.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_ops_n_z.h"
#include "machina/compiler/mlir/machina/ir/tf_saved_model.h"

namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_DEF_LEGALIZEVARIABLESPASS
#include "machina/compiler/mlir/lite/transforms/passes.h.inc"

// Attribute name to identify whether variables should be legalized to TFLite or
// not.
const char kLegalizeTflVariables[] = "tfl._legalize_tfl_variables";

bool HasSupportedElementType(Operation* op) {
  return utils::IsSupportedVariableType(op);
}

bool IsSupportedElementType(ShapedType type) {
  return utils::IsSupportedVariableType(type);
}

#include "machina/compiler/mlir/lite/transforms/generated_legalize_variables.inc"

// Pass which legalizes TF variables which are already passed as bounded
// arguments to functions, to TFLite variables.
class LegalizeVariablesPass
    : public impl::LegalizeVariablesPassBase<LegalizeVariablesPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeVariablesPass)

  void runOnOperation() override {
    auto module = getOperation();
    // If TFLite variable legalization is not allowed, then we skip this pass.
    if (auto legalize_tfl_variables_attr =
            module->getAttr(kLegalizeTflVariables)) {
      if (!mlir::cast<BoolAttr>(legalize_tfl_variables_attr).getValue()) return;
    }

    RewritePatternSet patterns(&getContext());
    populateWithGenerated(patterns);
    (void)applyPatternsGreedily(module, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeVariablesPass() {
  return std::make_unique<LegalizeVariablesPass>();
}

}  // namespace TFL
}  // namespace mlir
