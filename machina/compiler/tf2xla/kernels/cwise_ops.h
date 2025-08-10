/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

// XLA-specific base classes for Unary and Binary Ops.

#ifndef MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_KERNELS_CWISE_OPS_H_
#define MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_KERNELS_CWISE_OPS_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/xla/client/client_library.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/util/bcast.h"

namespace machina {

// Coefficient-wise binary operations. Each binary Op expects two
// inputs that can be broadcast to the same shape. The base class
// contains pure virtual methods to override: description is a textual
// description of the operation; and Computation adds the
// implementation of the operation to a xla::XlaBuilder. For most
// arithmetic Ops XLA handles the broadcasting automatically given the input
// tensors.
class XlaBinaryOp : public XlaOpKernel {
 public:
  explicit XlaBinaryOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    const DataType lhs = BaseType(input_type(0));
    const DataType rhs = BaseType(input_type(1));
    OP_REQUIRES(ctx, lhs == rhs,
                errors::InvalidArgument("Input types of binary op must match"));
  }
  ~XlaBinaryOp() override = default;

  // Implement the (tensor,tensor)->tensor lambda that should be
  // applied to the inputs. The desired computation should be added to
  // 'tc->builder()' and '(lhs,rhs)' are the function's inputs and
  // (lhs_shape,rhs_shape) are their respective
  // shapes. 'broadcast_helper' contains metadata about the shapes of
  // the inputs and the dimensions that need to be broadcast, which
  // may be useful for Ops that can't use standard XLA automatic
  // broadcasting. 'extend_dimension' is non-empty if lhs and rhs have
  // different ranks, and indicates which dimensions of the
  // higher-rank input should be matched when broadcasting the
  // lower-rank input. See comment below and the documentation on broadcasting
  // in the XLA documentation.
  virtual xla::XlaOp Computation(
      XlaOpKernelContext* ctx, const xla::XlaOp& lhs,
      const absl::Span<const int64_t>& lhs_shape, const xla::XlaOp& rhs,
      const absl::Span<const int64_t>& rhs_shape, const BCast& broadcast_helper,
      const std::vector<int64_t>& extend_dimensions) = 0;

  void Compile(XlaOpKernelContext* ctx) override;

  // Helper function that performs the broadcasting described by
  // 'broadcast_helper', yielding arguments 'lhs' and 'rhs' that have the same
  // shape.
  static std::pair<xla::XlaOp, xla::XlaOp> Broadcast(
      xla::XlaOp lhs, xla::XlaOp rhs, const BCast& broadcast_helper);
};

}  // namespace machina

#endif  // MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_KERNELS_CWISE_OPS_H_
