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

#include "toolchain/Support/Casting.h"
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"  // IWYU pragma: keep

namespace mlir {
namespace odml {
#define GEN_PASS_DEF_LIFTCALLSITELOCCALLERPASS
#include "machina/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h.inc"

namespace {

// JAX bridge generates a func.call for each op lowering
// These are inlined but loc will be messed up after the inline pass. This pass
// normalize the loc after inline pass.

class LiftCallSiteLocCallerPass
    : public impl::LiftCallSiteLocCallerPassBase<LiftCallSiteLocCallerPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LiftCallSiteLocCallerPass);

  void runOnOperation() override {
    getOperation()->walk([](mlir::Operation* op) {
      while (true) {
        auto loc = mlir::dyn_cast_or_null<CallSiteLoc>(op->getLoc());
        if (loc == nullptr) {
          return;
        }

        if (toolchain::isa<mlir::UnknownLoc>(loc.getCallee())) {
          op->setLoc(loc.getCaller());
        } else {
          op->setLoc(loc.getCallee());
        }
      }
    });
  }
};

}  // namespace
}  // namespace odml
}  // namespace mlir
