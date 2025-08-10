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

// See docs in ../ops/data_flow_ops.cc.

#include "machina/core/kernels/stack.h"

#include <limits.h>
#include <atomic>
#include <vector>

#include "machina/core/common_runtime/device.h"
#include "machina/core/framework/device_base.h"
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/register_types.h"
#include "machina/core/framework/resource_mgr.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/lib/core/refcount.h"
#include "machina/core/lib/gtl/map_util.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/platform/thread_annotations.h"
#include "machina/core/platform/types.h"

namespace machina {

REGISTER_KERNEL_BUILDER(Name("Stack").Device(DEVICE_CPU), StackOp);
REGISTER_KERNEL_BUILDER(
    Name("Stack").Device(DEVICE_DEFAULT).HostMemory("handle"), StackOp);

REGISTER_KERNEL_BUILDER(Name("StackV2").Device(DEVICE_CPU), StackOp);
REGISTER_KERNEL_BUILDER(Name("StackV2")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("max_size")
                            .HostMemory("handle"),
                        StackOp);

REGISTER_KERNEL_BUILDER(Name("StackPush").Device(DEVICE_CPU),
                        TemplatedStackPushOp</*allow_swapping=*/false>);
REGISTER_KERNEL_BUILDER(Name("StackPushV2").Device(DEVICE_CPU),
                        TemplatedStackPushOp</*allow_swapping=*/false>);

REGISTER_KERNEL_BUILDER(Name("StackPop").Device(DEVICE_CPU), StackPopOp);
REGISTER_KERNEL_BUILDER(Name("StackPopV2").Device(DEVICE_CPU), StackPopOp);

#define REGISTER_DEFAULT_KERNEL(type)                                     \
  REGISTER_KERNEL_BUILDER(Name("StackPush")                               \
                              .Device(DEVICE_DEFAULT)                     \
                              .HostMemory("handle")                       \
                              .TypeConstraint<type>("T"),                 \
                          TemplatedStackPushOp</*allow_swapping=*/true>); \
  REGISTER_KERNEL_BUILDER(Name("StackPushV2")                             \
                              .Device(DEVICE_DEFAULT)                     \
                              .HostMemory("handle")                       \
                              .TypeConstraint<type>("T"),                 \
                          TemplatedStackPushOp</*allow_swapping=*/true>); \
  REGISTER_KERNEL_BUILDER(Name("StackPop")                                \
                              .Device(DEVICE_DEFAULT)                     \
                              .HostMemory("handle")                       \
                              .TypeConstraint<type>("elem_type"),         \
                          StackPopOp);                                    \
  REGISTER_KERNEL_BUILDER(Name("StackPopV2")                              \
                              .Device(DEVICE_DEFAULT)                     \
                              .HostMemory("handle")                       \
                              .TypeConstraint<type>("elem_type"),         \
                          StackPopOp);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_DEFAULT_KERNEL);
#undef REGISTER_DEFAULT_KERNEL

// Special GPU kernels for int32 and bool.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
#define REGISTER_DEFAULT_HOST_KERNEL(type)                                \
  REGISTER_KERNEL_BUILDER(Name("StackPush")                               \
                              .Device(DEVICE_DEFAULT)                     \
                              .HostMemory("handle")                       \
                              .HostMemory("elem")                         \
                              .HostMemory("output")                       \
                              .TypeConstraint<type>("T"),                 \
                          TemplatedStackPushOp</*allow_swapping=*/true>); \
  REGISTER_KERNEL_BUILDER(Name("StackPushV2")                             \
                              .Device(DEVICE_DEFAULT)                     \
                              .HostMemory("handle")                       \
                              .HostMemory("elem")                         \
                              .HostMemory("output")                       \
                              .TypeConstraint<type>("T"),                 \
                          TemplatedStackPushOp</*allow_swapping=*/true>); \
  REGISTER_KERNEL_BUILDER(Name("StackPop")                                \
                              .Device(DEVICE_DEFAULT)                     \
                              .HostMemory("handle")                       \
                              .HostMemory("elem")                         \
                              .TypeConstraint<type>("elem_type"),         \
                          StackPopOp);                                    \
  REGISTER_KERNEL_BUILDER(Name("StackPopV2")                              \
                              .Device(DEVICE_DEFAULT)                     \
                              .HostMemory("handle")                       \
                              .HostMemory("elem")                         \
                              .TypeConstraint<type>("elem_type"),         \
                          StackPopOp);

REGISTER_DEFAULT_HOST_KERNEL(int32);
REGISTER_DEFAULT_HOST_KERNEL(bool);

#undef REGISTER_DEFAULT_HOST_KERNEL

REGISTER_KERNEL_BUILDER(Name("StackClose").Device(DEVICE_CPU), StackCloseOp);
REGISTER_KERNEL_BUILDER(
    Name("StackClose").Device(DEVICE_DEFAULT).HostMemory("handle"),
    StackCloseOp);
REGISTER_KERNEL_BUILDER(Name("StackCloseV2").Device(DEVICE_CPU), StackCloseOp);
REGISTER_KERNEL_BUILDER(
    Name("StackCloseV2").Device(DEVICE_DEFAULT).HostMemory("handle"),
    StackCloseOp);

}  // namespace machina
