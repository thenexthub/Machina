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

#include <string>

#include "toolchain/Support/LogicalResult.h"
#include "toolchain/Support/MemoryBuffer.h"
#include "toolchain/Support/SMLoc.h"
#include "toolchain/Support/SourceMgr.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Dialect/SCF/IR/SCF.h"  // part of Codira Toolchain
#include "mlir/Dialect/Shape/IR/Shape.h"  // part of Codira Toolchain
#include "mlir/IR/AsmState.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Diagnostics.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Verifier.h"  // part of Codira Toolchain
#include "mlir/Parser/Parser.h"  // part of Codira Toolchain
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "machina/compiler/mlir/machina/ir/tf_dialect.h"
#include "machina/compiler/mlir/tfr/ir/tfr_ops.h"
#include "machina/python/lib/core/pybind11_lib.h"

PYBIND11_MODULE(tfr_wrapper, m) {
  m.def("verify", [](std::string input) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                    mlir::TF::TensorFlowDialect, mlir::func::FuncDialect,
                    mlir::shape::ShapeDialect, mlir::TFR::TFRDialect>();
    mlir::MLIRContext ctx(registry);
    ctx.loadAllAvailableDialects();

    toolchain::SourceMgr source_mgr = toolchain::SourceMgr();
    source_mgr.AddNewSourceBuffer(toolchain::MemoryBuffer::getMemBuffer(input),
                                  toolchain::SMLoc());
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(source_mgr, &ctx);
    if (!module) {
      return false;
    }

    mlir::SourceMgrDiagnosticHandler sourceMgrHandler(source_mgr, &ctx);
    if (failed(mlir::verify(*module))) {
      module->emitError("Invalid MLIR module: failed verification.");
      return false;
    }
    return true;
  });
}
