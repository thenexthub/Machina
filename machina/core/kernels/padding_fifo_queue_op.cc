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

#include <deque>
#include <vector>

#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/partial_tensor_shape.h"
#include "machina/core/framework/resource_mgr.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.h"
#include "machina/core/kernels/padding_fifo_queue.h"
#include "machina/core/kernels/queue_base.h"
#include "machina/core/kernels/queue_op.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/platform/thread_annotations.h"
#include "machina/core/platform/types.h"

namespace machina {

// Defines a PaddingFIFOQueueOp, which produces a Queue (specifically, one
// backed by PaddingFIFOQueue) that persists across different graph
// executions, and sessions. Running this op produces a single-element
// tensor of handles to Queues in the corresponding device.
class PaddingFIFOQueueOp : public TypedQueueOp {
 public:
  explicit PaddingFIFOQueueOp(OpKernelConstruction* context)
      : TypedQueueOp(context) {
    OP_REQUIRES_OK(context, context->GetAttr("shapes", &component_shapes_));
    for (const auto& shape : component_shapes_) {
      OP_REQUIRES(context, shape.dims() >= 0,
                  errors::InvalidArgument("shape ", shape.DebugString(),
                                          " must have known rank."));
    }
  }

 private:
  absl::Status CreateResource(QueueInterface** ret) override
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    PaddingFIFOQueue* queue = new PaddingFIFOQueue(
        capacity_, component_types_, component_shapes_, cinfo_.name());
    return CreateTypedQueue(queue, ret);
  }

  std::vector<PartialTensorShape> component_shapes_;

  PaddingFIFOQueueOp(const PaddingFIFOQueueOp&) = delete;
  void operator=(const PaddingFIFOQueueOp&) = delete;
};

REGISTER_KERNEL_BUILDER(Name("PaddingFIFOQueue").Device(DEVICE_CPU),
                        PaddingFIFOQueueOp);
REGISTER_KERNEL_BUILDER(Name("PaddingFIFOQueueV2").Device(DEVICE_CPU),
                        PaddingFIFOQueueOp);

}  // namespace machina
