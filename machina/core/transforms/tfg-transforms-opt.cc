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

#include "toolchain/Support/LogicalResult.h"
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/Tools/mlir-opt/MlirOptMain.h"  // part of Codira Toolchain
#include "mlir/Transforms/Passes.h"  // part of Codira Toolchain
#include "machina/core/ir/dialect.h"
#include "machina/core/ir/tf_op_registry.h"
#include "machina/core/ir/types/dialect.h"
#include "machina/core/transforms/pass_registration.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerCanonicalizerPass();
  mlir::registerPrintOpStatsPass();
  mlir::registerViewOpGraphPass();
  mlir::registerSymbolDCEPass();
  mlir::registerSymbolPrivatizePass();
  mlir::tfg::registerTFGraphPasses();
  registry.insert<mlir::tfg::TFGraphDialect, mlir::tf_type::TFTypeDialect>();
  // Inject the op registry.
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, mlir::tfg::TFGraphDialect *dialect) {
        dialect->addInterfaces<mlir::tfg::TensorFlowOpRegistryInterface>();
      });
  return failed(
      mlir::MlirOptMain(argc, argv, "TFGraph Transforms Driver", registry));
}
