/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

// This file implements logic for translating mixed IR to buffer form.

#include <memory>

#include "mlir/Dialect/Arith/IR/Arith.h"  // part of Codira Toolchain
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "machina/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"
#include "machina/xla/mlir_hlo/transforms/passes.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

#define GEN_PASS_DEF_KERNELGENFINALBUFFERIZEPASS
#include "machina/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

struct KernelgenFinalBufferizePass
    : public impl::KernelgenFinalBufferizePassBase<
          KernelgenFinalBufferizePass> {
  // Default alignment_ specified in passes.td
  KernelgenFinalBufferizePass() = default;

  void runOnOperation() override {
    mlir::PassManager pm(&getContext());
    pm.addPass(mlir::createFinalBufferizePass(/*alignment=*/64,
                                              populateExtraBufferizeDialects,
                                              populateExtraBufferizePatterns));
    (void)runPipeline(pm, getOperation());
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateKernelgenFinalBufferizePass() {
  return std::make_unique<KernelgenFinalBufferizePass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
