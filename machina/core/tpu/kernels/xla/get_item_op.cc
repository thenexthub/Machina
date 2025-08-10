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

#include <cstdint>
#include <vector>

#define EIGEN_USE_THREADS

#include "machina/compiler/tf2xla/xla_op_kernel.h"
#include "machina/compiler/tf2xla/xla_op_registry.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/op_requires.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/platform/errors.h"

namespace machina {
namespace {

// The Xla kernel to build up the computation for get_item(data, index).
class GetItemXlaOp : public XlaOpKernel {
 public:
  explicit GetItemXlaOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape& data_shape = ctx->InputShape(0);
    const TensorShape& index_shape = ctx->InputShape(1);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVectorOrHigher(data_shape),
        errors::InvalidArgument("data must be at least 1 dimensional."));
    OP_REQUIRES(ctx, index_shape.dims() == 1 && index_shape.dim_size(0) == 1,
                errors::InvalidArgument("index must be a vector of size 1."));

    // NOTE(pbar) Use Concat to extend the indices to match cl/142279605.
    // This isn't the simplest way to emit the indices, but the code for
    // dynamic slice needs to be able to see that minor dims are const zero.
    auto const_zero = xla::ConstantR0(ctx->builder(), 0);
    std::vector<xla::XlaOp> operands;
    operands.push_back(xla::Reshape(ctx->Input(1), {}));
    for (int i = 1; i < data_shape.dims(); i++) {
      operands.push_back(const_zero);
    }

    std::vector<int64_t> slice_sizes = {1};
    std::vector<int64_t> out_sizes = {};
    for (int i = 1; i < data_shape.dims(); i++) {
      auto size = data_shape.dim_size(i);
      slice_sizes.push_back(size);
      out_sizes.push_back(size);
    }
    // NOTE: DynamicSlice here doesn't raise an error or wraps the index
    // if its out-of-range.
    auto slice = xla::DynamicSlice(ctx->Input(0), operands, slice_sizes);
    // In-order collapse to remove the 1st dim.
    auto reshape = xla::Reshape(slice, out_sizes);
    ctx->SetOutput(0, reshape);
  }
};

REGISTER_MACHINA_MACHINA_XLA_OP(Name("GetItem"), GetItemXlaOp);

}  // namespace
}  // namespace machina
