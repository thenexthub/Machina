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

// See docs in ../ops/io_ops.cc.
#include "machina/core/kernels/save_restore_tensor.h"

#include "machina/core/framework/op_kernel.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/lib/gtl/array_slice.h"
#include "machina/core/platform/logging.h"
#include "machina/core/util/tensor_slice_reader.h"

namespace machina {

class RestoreOp : public OpKernel {
 public:
  explicit RestoreOp(OpKernelConstruction* context) : OpKernel(context) {
    int preferred_shard;
    OP_REQUIRES_OK(context,
                   context->GetAttr("preferred_shard", &preferred_shard));
    if (preferred_shard == -1) {
      preferred_shard_ = checkpoint::TensorSliceReader::kLoadAllShards;
    } else {
      OP_REQUIRES(context, preferred_shard >= 0,
                  errors::InvalidArgument("Attribute 'preferred_shard' must be "
                                          "greater or equal to -1"));
      preferred_shard_ = preferred_shard;
    }
  }
  void Compute(OpKernelContext* context) override {
    RestoreTensor(context, &checkpoint::OpenTableTensorSliceReader,
                  preferred_shard_, false, 0);
  }

 private:
  int preferred_shard_;
};

REGISTER_KERNEL_BUILDER(Name("Restore").Device(DEVICE_CPU), RestoreOp);

class RestoreSliceOp : public OpKernel {
 public:
  explicit RestoreSliceOp(OpKernelConstruction* context) : OpKernel(context) {
    int preferred_shard;
    OP_REQUIRES_OK(context,
                   context->GetAttr("preferred_shard", &preferred_shard));
    if (preferred_shard == -1) {
      preferred_shard_ = checkpoint::TensorSliceReader::kLoadAllShards;
    } else {
      OP_REQUIRES(context, preferred_shard >= 0,
                  errors::InvalidArgument("Attribute 'preferred_shard' must be "
                                          "greater or equal to -1"));
      preferred_shard_ = preferred_shard;
    }
  }
  void Compute(OpKernelContext* context) override {
    RestoreTensor(context, &checkpoint::OpenTableTensorSliceReader,
                  preferred_shard_, true, 0);
  }

 private:
  int preferred_shard_;
};

REGISTER_KERNEL_BUILDER(Name("RestoreSlice").Device(DEVICE_CPU),
                        RestoreSliceOp);

}  // namespace machina
