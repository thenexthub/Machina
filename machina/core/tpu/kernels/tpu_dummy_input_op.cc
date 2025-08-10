/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, August 10, 2025.
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

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/op_requires.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.h"
#include "machina/core/framework/types.pb.h"
#include "tsl/platform/bfloat16.h"

namespace machina {

namespace {

class TpuDummyInputOp : public OpKernel {
 public:
  explicit TpuDummyInputOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &shape_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape_, &output));
    if (dtype_ == DT_FLOAT) {
      output->flat<float>().setZero();
    } else if (dtype_ == DT_BFLOAT16) {
      output->flat<tsl::bfloat16>().setZero();
    } else {
      ctx->SetStatus(absl::InternalError(
          absl::StrCat("Unsupported dtype: ", DataTypeString(dtype_))));
      return;
    }
  }

 private:
  DataType dtype_;
  TensorShape shape_;
};

// TODO(mrry): Add registration for TPU.
REGISTER_KERNEL_BUILDER(Name("TPUDummyInput").Device(DEVICE_CPU),
                        TpuDummyInputOp);

}  // namespace

}  // namespace machina
