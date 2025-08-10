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

#ifndef MACHINA_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_COMPOSITE_AVG_POOL_H_
#define MACHINA_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_COMPOSITE_AVG_POOL_H_

#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypeInterfaces.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/lite/transforms/passes.h"  // IWYU pragma: keep
#include "machina/xla/mlir_hlo/mhlo/IR/hlo_ops.h"  // IWYU pragma: keep

namespace mlir {
namespace odml {

// Given a Composite op that wraps a core.aten.avg_pool2d, returns the padding
// configuration required for the `tfl.pad` if the padding part of the op is
// to be done before average pooling.
DenseIntElementsAttr GetPadOpAttr(Builder& builder, mhlo::CompositeOp op);

// Given a Composite op that wraps a core.aten.avg_pool2d, and assuming that
// the padding part is extracted into a tfl.pad op prior to a
// tfl.average_pool_2d, this function finds the return type of the needed
// tfl.pad .
ShapedType GetPadOpType(mhlo::CompositeOp op);

// Given a Composite op that wraps a core.aten.avg_pool2d, finds the padding
// attribute to be passed to the a tfl.average_pool_2d that can fully replace
// this composite (here, padding is done directly by the tfl.average_pool_2d as
// opposed to being extracted into a separate tfl.pad).
StringAttr GetAvgPoolOpPadAttr(Builder& builder, mhlo::CompositeOp op);

// Get dense attr for a matrix that corrects the over counting of divisors when
// casting an average pool with ceil mode on in terms of average pool with it
// off.
DenseFPElementsAttr GetCorrectionMatrix(Builder& builder, mhlo::CompositeOp op);

}  // namespace odml
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_COMPOSITE_AVG_POOL_H_
