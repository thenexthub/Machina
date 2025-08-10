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
#include <deque>

#include "machina/core/common_runtime/process_function_library_runtime.h"
#include "machina/core/framework/dataset.h"
#include "machina/core/framework/function.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/resource_op_kernel.h"
#include "machina/core/lib/core/threadpool.h"
#include "machina/core/lib/random/random.h"
#include "machina/core/util/device_name_utils.h"

namespace machina {
namespace data {
namespace experimental {
namespace {

class IteratorGetDeviceOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* ctx) override {
    // NOTE(mrry): We do not currently Validate that the handle
    // corresponds to a real IteratorResource, because that symbol is
    // not exposed from the framework library.
    Tensor* device_name_t;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({}), &device_name_t));
    // NOTE(mrry): Since the operation's input is a resource, we must be
    // colocated with it, and so we can simply return the current device's
    // name without looking at the input.
    device_name_t->scalar<tstring>()() = ctx->device()->name();
  }
};

REGISTER_KERNEL_BUILDER(Name("IteratorGetDevice").Device(DEVICE_CPU),
                        IteratorGetDeviceOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalIteratorGetDevice").Device(DEVICE_CPU),
    IteratorGetDeviceOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace machina
