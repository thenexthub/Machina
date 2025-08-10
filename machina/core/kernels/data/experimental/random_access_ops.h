/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#ifndef MACHINA_CORE_KERNELS_DATA_EXPERIMENTAL_RANDOM_ACCESS_OPS_H_
#define MACHINA_CORE_KERNELS_DATA_EXPERIMENTAL_RANDOM_ACCESS_OPS_H_

#include "machina/core/framework/dataset.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/op_requires.h"
#include "machina/core/kernels/data/iterator_ops.h"
#include "machina/core/platform/platform.h"

namespace machina {
namespace data {
namespace experimental {

// An operation that can get an element at a specified index in a dataset.
class GetElementAtIndexOp : public AsyncOpKernel {
 public:
  explicit GetElementAtIndexOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        unbounded_threadpool_(ctx->env(), "tf_data_get_element_at_index") {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    unbounded_threadpool_.Schedule([this, ctx, done = std::move(done)]() {
      ctx->SetStatus(DoCompute(ctx));
      done();
    });
  }

  void Compute(OpKernelContext* ctx) override {
    ctx->SetStatus(DoCompute(ctx));
  }

 protected:
  absl::Status DoCompute(OpKernelContext* ctx);

 private:
  UnboundedThreadPool unbounded_threadpool_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

}  // namespace experimental
}  // namespace data
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_DATA_EXPERIMENTAL_RANDOM_ACCESS_OPS_H_
