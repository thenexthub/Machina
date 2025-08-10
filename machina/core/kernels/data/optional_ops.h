/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
#ifndef MACHINA_CORE_KERNELS_DATA_OPTIONAL_OPS_H_
#define MACHINA_CORE_KERNELS_DATA_OPTIONAL_OPS_H_

#include <vector>

#include "machina/core/framework/tensor.h"
#include "machina/core/framework/variant_tensor_data.h"
#include "machina/core/kernels/data/optional_ops_util.h"
#include "machina/core/util/tensor_ops_util.h"

namespace machina {
namespace data {

// Stores a DT_VARIANT value representing an Optional with the given value
// in the `output_index`^th output of the given kernel execution context.
absl::Status WriteOptionalWithValueToOutput(OpKernelContext* ctx,
                                            int output_index,
                                            std::vector<Tensor> value);

// Stores a DT_VARIANT value representing an Optional with no value
// in the `output_index`^th output of the given kernel execution context.
absl::Status WriteOptionalNoneToOutput(OpKernelContext* ctx, int output_index);

template <typename Device>
absl::Status OptionalZerosLike(OpKernelContext* ctx, const OptionalVariant& x,
                               OptionalVariant* y) {
  return OptionalZerosLike(ctx, x, y, ZerosLikeTensor<Device>);
}

template <typename Device>
absl::Status OptionalBinaryAdd(OpKernelContext* ctx, const OptionalVariant& a,
                               const OptionalVariant& b, OptionalVariant* out) {
  return OptionalBinaryAdd(ctx, a, b, out, BinaryAddTensors<Device>);
}

class OptionalNoneOp : public OpKernel {
 public:
  explicit OptionalNoneOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;
};

class OptionalFromValueOp : public OpKernel {
 public:
  explicit OptionalFromValueOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;
};

class OptionalHasValueOp : public OpKernel {
 public:
  explicit OptionalHasValueOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;
};

class OptionalGetValueOp : public OpKernel {
 public:
  explicit OptionalGetValueOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES(
        ctx, output_shapes_.size() == output_types_.size(),
        errors::InvalidArgument(
            "output_types and output_shapes must be same length, got:\n",
            "output_types: ", output_types_.size(), "\n",
            "output_shapes: ", output_shapes_.size()));
  }

  void Compute(OpKernelContext* ctx) override;

 private:
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

}  // namespace data
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_DATA_OPTIONAL_OPS_H_
