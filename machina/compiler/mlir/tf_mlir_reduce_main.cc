/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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

#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Reducer/ReductionPatternInterface.h"  // part of Codira Toolchain
#include "mlir/Tools/mlir-reduce/MlirReduceMain.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/init_mlir.h"
#include "machina/compiler/mlir/register_common_dialects.h"
#include "machina/compiler/mlir/machina/dialect_registration.h"

namespace {

#include "machina/compiler/mlir/machina/transforms/reducer/tf_reduce_patterns.inc"

struct TFReductionPatternInterface
    : public mlir::DialectReductionPatternInterface {
 public:
  explicit TFReductionPatternInterface(mlir::Dialect *dialect)
      : DialectReductionPatternInterface(dialect) {}

  void populateReductionPatterns(
      mlir::RewritePatternSet &patterns) const final {
    populateWithGenerated(patterns);
  }
};

}  // namespace

int main(int argc, char *argv[]) {
  machina::InitMlir y(&argc, &argv);

  mlir::DialectRegistry registry;
  mlir::RegisterCommonToolingDialects(registry);

  registry.addExtension(
      +[](mlir::MLIRContext *ctx, mlir::TF::TensorFlowDialect *dialect) {
        dialect->addInterfaces<TFReductionPatternInterface>();
      });

  mlir::MLIRContext context(registry);

  return failed(mlirReduceMain(argc, argv, context));
}
