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

#ifndef MACHINA_DTENSOR_MLIR_COLLECTIVES_H_
#define MACHINA_DTENSOR_MLIR_COLLECTIVES_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_device.h"
#include "machina/core/platform/types.h"
#include "machina/dtensor/cc/dstatus.h"
#include "machina/dtensor/cc/tensor_layout.h"

namespace machina {
namespace dtensor {

// Emits collective ops to convert `input` from `src_layout` to `tgt_layout`.
// `src_layout` and `tgt_layout` must have the same rank. For each dimension,
// it can only go from sharded to replicated. `input` must have static shapes.
StatusOr<mlir::Value> EmitAllGather(
    mlir::OpBuilder& builder, mlir::Value input,
    const dtensor::Layout& src_layout, const dtensor::Layout& tgt_layout,
    toolchain::SmallPtrSet<mlir::Operation*, 4>* newly_created_ops = nullptr);

// Given an input layout and a desired layout, inserts the necessary slice to
// slice the original value based on the device id. All ops created by this
// function are added to new_created_ops.
//
// Note that the newly created ops are inserted `after` original_value.
StatusOr<const mlir::Value> EmitAllScatter(
    mlir::OpBuilder& builder, const mlir::Value& original_value,
    const Layout& original_layout, const Layout& desired_layout,
    toolchain::SmallPtrSet<mlir::Operation*, 4>* newly_created_ops = nullptr);

// Emits splits and calls EmitAllGather (once) to relayout from the src layout
// to the tgt layout on a single mesh.
// Shape of input is expected to be the local shape for src_layout.
StatusOr<mlir::Value> EmitRelayout(
    mlir::Value input, const dtensor::Layout& src_layout,
    const dtensor::Layout& tgt_layout,
    toolchain::SmallPtrSet<mlir::Operation*, 4>* newly_created_ops = nullptr);

// Emits TransposeOp that permutes the input shape.
mlir::Operation* EmitTransposeOp(mlir::OpBuilder& builder,
                                 const mlir::Location& loc, mlir::Value input,
                                 std::vector<int64_t>& perm_arr);

// Emits collective ops to reduce `input` over `reduced_dims`.
StatusOr<mlir::Operation*> EmitAllReduce(
    mlir::OpBuilder& builder, const dtensor::Layout& output_layout,
    const absl::flat_hash_set<std::string>& reduced_dims,
    mlir::Operation* input, absl::string_view reduce_op);

// Emits a barrier used for synchronization purposes and returns
// a R1 const value using `value`. More precisely, this barrier
// guarantees that
//    1. Side-effect Ops before this barrier are complete before this op begins.
//    2. Side-effect Ops after this barrier start after this barrier completes.
//
// Note that the returned operation must be used in the graph. If it is not
// used, then this op will be removed from the graph from various compiler
// passes and thus there will be no barrier.
//
// Used for introducing a barrier before every Merge op during checkpointing.
StatusOr<mlir::Operation*> EmitBarrierWithConstValue(mlir::OpBuilder& builder,
                                                     mlir::Location loc,
                                                     const Mesh& mesh,
                                                     int32 value);

// Given input `tensor` that is sharded across spatial dimensions, conduct
// halo exchange such that each spatially sharded input blocks exchange
// `halo_size` slice with its neighboring processors.
// If the input block is at the left/right/top/bottom edge, then ghost halo
// tensor (zero) are padded instead. `mesh_dim` specifies the dimension which
// halo exchange will be conducted. For example, if we consider a 4D Tensor
// (batch, height, width, channel) that has layout (*, h, w, *). Then,
// `mesh_dim` ==  "w" would mean that halo exchange will occur along the width
// dimension. That is halo tensors with right/left neighbors will be exchanged.
StatusOr<mlir::Value> EmitHaloExchange(mlir::OpBuilder& builder, int halo_size,
                                       const std::string& mesh_dim,
                                       const Layout& layout,
                                       mlir::Value mesh_coordinates,
                                       mlir::tf_device::ClusterOp cluster,
                                       mlir::Location location,
                                       mlir::Value tensor);

// Emits a DenseToSparse op followed by a SparseToDenseOp.
// This is useful for emitting a Relayout on a SparseTensor.
// One usage of this is in EmitRelayout when the input is a SparseTensor.
StatusOr<mlir::Value> EmitDenseToSparseToDense(
    mlir::OpBuilder& builder, mlir::Value input,
    toolchain::SmallPtrSet<mlir::Operation*, 4>* newly_created_ops = nullptr);

}  // namespace dtensor
}  // namespace machina

#endif  // MACHINA_DTENSOR_MLIR_COLLECTIVES_H_
