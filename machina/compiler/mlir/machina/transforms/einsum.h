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
// This pass identifies patterns for certain Einsum Ops and replaces them
// with other equivalent TF Ops.

#ifndef MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_EINSUM_H_
#define MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_EINSUM_H_

#include <cstdint>
#include <initializer_list>

#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/Matchers.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/IR/TypeUtilities.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/core/util/matmul_bcast.h"

namespace mlir {
namespace TF {

// TF.Einsum provides fully general tensor contractions. For a few select
// cases, we can convert this op to other TF Ops, which in later passes
// properly convert to TF Lite ops.
struct ConvertTFEinsumOp : public OpRewritePattern<TF::EinsumOp> {
 public:
  explicit ConvertTFEinsumOp(MLIRContext* context)
      : OpRewritePattern<TF::EinsumOp>(context) {}

  LogicalResult matchAndRewrite(TF::EinsumOp op,
                                PatternRewriter& rewriter) const override;
};

}  // namespace TF
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_EINSUM_H_
