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
#ifndef MACHINA_COMPILER_MLIR_TFR_INTEGRATION_TFR_DECOMPOSE_CTX_H_
#define MACHINA_COMPILER_MLIR_TFR_INTEGRATION_TFR_DECOMPOSE_CTX_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/translate/mlir_roundtrip_flags.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/stringpiece.h"

namespace machina {
namespace tfr {

extern const char* const kTFRLibEnv;

using tsl::StatusOr;

// An wrapper for all the objects used to decompose a module (graph mode) and
// node_def (eager mode). Note that this class owns the decomposition library.
class TFRDecomposeContext {
 public:
  // The entry function to get a decompose context. All the required passes have
  // been initialized.
  static absl::StatusOr<std::unique_ptr<TFRDecomposeContext>> Get(
      mlir::MLIRContext* mlir_ctx);

  // Constructor of the decompose context. To share the decompose library, the
  // whole decompose TFR function library is loaded.
  explicit TFRDecomposeContext(mlir::ModuleOp tfr_module);

  // Constructs the decompose context from the tfr text module and the mlir
  // context. The tfr text module is added to the mlir context.
  static std::unique_ptr<TFRDecomposeContext> GetFromText(
      absl::string_view tfr_raw_text, mlir::MLIRContext* mlir_ctx);

  // Decomposes the op in the NodeDef to a set of primitive ops according to the
  // decompose library in the context. Wrap the decomposed result in a
  // FunctionDef.
  absl::StatusOr<FunctionDef> ExpandNode(const NodeDef& node_def,
                                         absl::string_view func_name);

  // Runs the decompose passes on the user_module.
  absl::Status DecomposeGraph(mlir::ModuleOp user_module);

  // Erases the tfr_module created.
  void Destroy();

 private:
  mlir::ModuleOp tfr_module_;
  mlir::PassManager pm_;

  GraphExportConfig export_confs_;
};

// Decomposes the NodeDef to a set of primitive ops according to the decompose
// library loaded. Wrap the decomposed result in a FunctionDef.
absl::StatusOr<FunctionDef> ExpandNode(const NodeDef& node_def,
                                       absl::string_view func_name);

// Decomposes the ops in the ModuleOp to a set of primitive ops according to
// decompose library in the context.
absl::Status DecomposeGraph(mlir::ModuleOp user_module);

}  // namespace tfr
}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TFR_INTEGRATION_TFR_DECOMPOSE_CTX_H_
