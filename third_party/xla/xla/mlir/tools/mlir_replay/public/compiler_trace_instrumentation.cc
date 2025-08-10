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

#include "machina/xla/mlir/tools/mlir_replay/public/compiler_trace_instrumentation.h"

#include <string>

#include "absl/log/log.h"
#include "toolchain/Support/Casting.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "machina/xla/service/llvm_ir/llvm_util.h"
#include "machina/xla/tsl/platform/logging.h"

namespace mlir {
namespace interpreter {

void MlirCompilerTraceInstrumentation::runAfterPass(Pass* pass, Operation* op) {
  ModuleOp module = toolchain::dyn_cast<ModuleOp>(op);
  if (!module) {
    module = op->getParentOfType<mlir::ModuleOp>();
  }
  if (!module) {
    LOG(ERROR) << "Failed to find a ModuleOp: " << pass->getName().str() << ".";
    return;
  }

  auto* item = trace_.mutable_passes()->Add();
  item->set_after_pass(pass->getName().str());
  *item->mutable_mlir_module() = xla::llvm_ir::DumpToString(module);
}

}  // namespace interpreter
}  // namespace mlir
