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

#ifndef MACHINA_DTENSOR_MLIR_LAYOUT_PARSING_H_
#define MACHINA_DTENSOR_MLIR_LAYOUT_PARSING_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "toolchain/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "machina/core/platform/status.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/proto/layout.pb.h"

namespace machina {
namespace dtensor {

// Extracts `_layout` attribute from `op` and assert a single layout.
StatusOr<std::optional<Layout>> ExtractSingleLayoutFromOp(mlir::Operation* op);

// Extracts `_layout` attribute from `op`, and returns an error is the layout
// is missing.
StatusOr<Layout> ExtractRequiredSingleLayoutFromOp(mlir::Operation* op);

// Extracts `_layout` attribute from `op` and assert a single layout.
StatusOr<std::optional<Layout>> ExtractSingleLayoutFromOp(
    mlir::Operation* op, std::string attr_name);

// Extracts `_layout` attribute from `op`.
StatusOr<std::vector<std::optional<Layout>>> ExtractLayoutFromOp(
    mlir::Operation* op);

// Extracts `_layout` attribute from `op` and returns an error if any are
// missing.
StatusOr<std::vector<Layout>> ExtractRequiredLayoutFromOp(mlir::Operation* op);

// Extract and deserialize a tensor layout from `attr_name`.
StatusOr<std::vector<std::optional<Layout>>> ExtractLayoutFromOp(
    mlir::Operation* op, std::string attr_name);

// Extracts '_layout' attribute from `operand`.
StatusOr<std::optional<Layout>> ExtractLayoutFromOperand(mlir::Value operand);

// Extracts '_layout' attribute from `operand` and returns an error if missing.
StatusOr<Layout> ExtractRequiredLayoutFromOperand(mlir::Value operand);

// Extracts `_layout` attribute from `op`'s operands and returns an error if
// any are missing.
StatusOr<std::vector<Layout>> ExtractRequiredLayoutFromOperands(
    mlir::Operation* op);

// Set `_layout` attribute for op. For layouts without value, an empty string is
// used as place holder.
void SetLayoutOnOp(mlir::Operation* op, mlir::OpBuilder builder,
                   absl::Span<const absl::optional<Layout>> layouts);

void SetLayoutOnOp(mlir::Operation* op,
                   absl::Span<const absl::optional<Layout>> layouts);

void SetSingleLayoutOnOp(mlir::Operation* op, const Layout& layout);

// Extracts device mesh configuration from op's enclosing tf_device.Cluster op.
StatusOr<Mesh> ExtractDeviceMeshEnclosingCluster(mlir::Operation* op);

// Extracts device mesh configuration from op's `_mesh` attribute.
StatusOr<std::optional<Mesh>> ExtractDeviceMeshFromOp(mlir::Operation* op);

// Extracts default layout information from function return attribute.
StatusOr<std::optional<Layout>> ExtractLayoutFromFunctionReturnAttr(
    mlir::func::ReturnOp return_op, int return_index);

// Extract element layouts from the iterator resource operand of an op that uses
// that iterator (e.g. IteratorGetNext, OptionalGetValue, etc.). The layouts are
// extracted from the `tf._element_layouts` attribute of that resource tensor.
StatusOr<toolchain::SmallVector<Layout, 4>> ExtractElementLayoutsFromOperand(
    mlir::OpOperand& input_value);

}  // namespace dtensor
}  // namespace machina

#endif  // MACHINA_DTENSOR_MLIR_LAYOUT_PARSING_H_
