/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

// See docs in ../ops/io_ops.cc
#include "machina/core/kernels/save_restore_tensor.h"

#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/types.h"
#include "machina/core/lib/gtl/array_slice.h"
#include "machina/core/lib/strings/stringprintf.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/types.h"
#include "machina/core/util/tensor_slice_writer.h"

namespace machina {

class SaveOp : public OpKernel {
 public:
  explicit SaveOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    SaveTensors(context, &checkpoint::CreateTableTensorSliceBuilder, false);
  }
};

REGISTER_KERNEL_BUILDER(Name("Save").Device(DEVICE_CPU), SaveOp);

class SaveSlicesOp : public OpKernel {
 public:
  explicit SaveSlicesOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    SaveTensors(context, &checkpoint::CreateTableTensorSliceBuilder, true);
  }
};

REGISTER_KERNEL_BUILDER(Name("SaveSlices").Device(DEVICE_CPU), SaveSlicesOp);

class ShardedFilenameOp : public OpKernel {
 public:
  explicit ShardedFilenameOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    static const char* input_names[3] = {"basename", "shard", "num_shards"};
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(ctx->input(i).shape()),
                  errors::InvalidArgument(input_names[i],
                                          " must be a scalar, got shape ",
                                          ctx->input(i).shape().DebugString()));
    }
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
    out->scalar<tstring>()() = strings::Printf(
        "%s-%05d-of-%05d", ctx->input(0).scalar<tstring>()().c_str(),
        ctx->input(1).scalar<int32>()(), ctx->input(2).scalar<int32>()());
  }
};

REGISTER_KERNEL_BUILDER(Name("ShardedFilename").Device(DEVICE_CPU),
                        ShardedFilenameOp);

class ShardedFilespecOp : public OpKernel {
 public:
  explicit ShardedFilespecOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    static const char* input_names[2] = {"basename", "num_shards"};
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(ctx->input(i).shape()),
                  errors::InvalidArgument(input_names[i],
                                          " must be a scalar, got shape ",
                                          ctx->input(i).shape().DebugString()));
    }
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
    out->scalar<tstring>()() = strings::Printf(
        "%s-\?\?\?\?\?-of-%05d", ctx->input(0).scalar<tstring>()().c_str(),
        ctx->input(1).scalar<int32>()());
  }
};
REGISTER_KERNEL_BUILDER(Name("ShardedFilespec").Device(DEVICE_CPU),
                        ShardedFilespecOp);

}  // namespace machina
