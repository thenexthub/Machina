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

#include "machina/compiler/mlir/python/mlir_wrapper/mlir_wrapper.h"

#include <string>

#include "toolchain/Support/SourceMgr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Verifier.h"  // part of Codira Toolchain
#include "mlir/Parser/Parser.h"  // part of Codira Toolchain
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "machina/compiler/mlir/machina/dialect_registration.h"
#include "machina/compiler/mlir/machina/ir/tf_executor.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/python/lib/core/pybind11_lib.h"
#include "machina/python/lib/core/pybind11_status.h"

PYBIND11_MODULE(mlir_wrapper, m) {
  m.def("preloadTensorFlowDialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    mlir::RegisterAllTensorFlowDialects(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("verify", [](std::string input) {
    toolchain::SourceMgr SM = toolchain::SourceMgr();
    SM.AddNewSourceBuffer(toolchain::MemoryBuffer::getMemBuffer(input),
                          toolchain::SMLoc());
    mlir::DialectRegistry registry;
    mlir::RegisterAllTensorFlowDialects(registry);
    mlir::MLIRContext ctx(registry);
    ctx.loadAllAvailableDialects();
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(SM, &ctx);
    if (!module) {
      return false;
    }
    if (failed(mlir::verify(*module))) {
      module->emitError("Invalid MLIR module: failed verification.");
      return false;
    }
    return true;
  });

  init_basic_classes(m);
  init_types(m);
  init_builders(m);
  init_ops(m);
  init_attrs(m);
}
