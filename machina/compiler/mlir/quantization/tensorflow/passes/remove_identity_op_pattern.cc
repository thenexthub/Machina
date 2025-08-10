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
#include "machina/compiler/mlir/quantization/machina/passes/remove_identity_op_pattern.h"

#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"

namespace mlir {
namespace quant {

LogicalResult RemoveIdentity::matchAndRewrite(TF::IdentityOp identity,
                                              PatternRewriter &rewriter) const {
  for (Operation *user : identity->getUsers()) {
    // Replace the op with the input if output is only used by TF ops.
    // Currently this is more on the conservative side since we need to ensure
    // every consumer op to be a TF op before applying this pattern. We can
    // consider to revisit this in the future if this turns out to be too
    // restrictive.
    if (user->getDialect()->getNamespace() != "tf") {
      return failure();
    }
    // Identity ops of returning values might be helpful for some other
    // compilers, so avoid removing these Identity ops.
    if (user->hasTrait<OpTrait::IsTerminator>()) {
      return failure();
    }
  }

  rewriter.replaceOp(identity, identity.getInput());
  return success();
}

}  // namespace quant
}  // namespace mlir
