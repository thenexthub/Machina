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
#ifndef MACHINA_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TAC_IMPORTER_EXPORTER_H_
#define MACHINA_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TAC_IMPORTER_EXPORTER_H_

#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain

namespace mlir {
namespace TFL {
namespace tac {

// Interface for Importing program to TAC (Target Aware Conversion) Module.
// This class is an interface for importing program in TAC.
// See TacModule in how to register it with the module and use it.
class TacImporter {
 public:
  virtual ~TacImporter() = default;

  // Imports and returns the Module for the imported program.
  virtual absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> Import() = 0;
};

// Interface for exporting a module.
// Users should implement the interface for exporting the result from TAC
// in their preferred way.
// See TacModule in how to register it with the module and use it.
class TacExporter {
 public:
  virtual ~TacExporter() = default;

  // Imports and returns the Module for the imported program.
  virtual absl::Status Export(mlir::ModuleOp module) = 0;
};
}  // namespace tac
}  // namespace TFL
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TAC_IMPORTER_EXPORTER_H_
