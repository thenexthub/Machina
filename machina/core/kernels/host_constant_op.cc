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

#include "machina/core/kernels/host_constant_op.h"

#include "machina/core/framework/allocator.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/types.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/macros.h"

namespace machina {

_HostConstantOp::_HostConstantOp(OpKernelConstruction* ctx)
    : OpKernel(ctx), tensor_(ctx->output_type(0)) {
  const TensorProto* proto = nullptr;
  AllocatorAttributes alloc_attr;
  alloc_attr.set_on_host(true);
  OP_REQUIRES_OK(ctx, ctx->GetAttr("value", &proto));
  OP_REQUIRES_OK(
      ctx, ctx->device()->MakeTensorFromProto(*proto, alloc_attr, &tensor_));
  OP_REQUIRES(
      ctx, ctx->output_type(0) == tensor_.dtype(),
      errors::InvalidArgument("Type mismatch between value (",
                              DataTypeString(tensor_.dtype()), ") and dtype (",
                              DataTypeString(ctx->output_type(0)), ")"));
}

void _HostConstantOp::Compute(OpKernelContext* ctx) {
  ctx->set_output(0, tensor_);
}

// A special DEVICE_DEFAULT kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Const")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("output")
                            .TypeConstraint<int32>("dtype"),
                        _HostConstantOp);

// HostConst: forced to generate output on the host.
REGISTER_KERNEL_BUILDER(Name("HostConst").Device(DEVICE_CPU), _HostConstantOp);
REGISTER_KERNEL_BUILDER(
    Name("HostConst").Device(DEVICE_DEFAULT).HostMemory("output"),
    _HostConstantOp);

}  // end namespace machina

