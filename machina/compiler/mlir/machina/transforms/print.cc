/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#include <memory>

#include "toolchain/Support/Mutex.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/transforms/passes.h"

namespace mlir {
namespace TF {

namespace {

#define GEN_PASS_DEF_PRINTPASS
#include "machina/compiler/mlir/machina/transforms/tf_passes.h.inc"

class PrintPass : public impl::PrintPassBase<PrintPass> {
 public:
  explicit PrintPass(raw_ostream* os = nullptr);
  PrintPass(const PrintPass& other);
  void runOnOperation() override;

 private:
  toolchain::sys::SmartMutex<true> mutex_;
  raw_ostream* os_;
};

PrintPass::PrintPass(raw_ostream* os) {
  if (os) {
    os_ = os;
  } else {
    os_ = &toolchain::errs();
  }
}

PrintPass::PrintPass(const PrintPass& other) : PrintPass(other.os_) {}

void PrintPass::runOnOperation() {
  toolchain::sys::SmartScopedLock<true> instrumentationLock(mutex_);
  OpPrintingFlags flags =
      OpPrintingFlags().elideLargeElementsAttrs().enableDebugInfo(false);
  getOperation()->print(*os_, flags);
}
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreatePrintPass(raw_ostream* os) {
  return std::make_unique<PrintPass>(os);
}

}  // namespace TF
}  // namespace mlir
