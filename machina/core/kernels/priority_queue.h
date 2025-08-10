/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#ifndef MACHINA_CORE_KERNELS_PRIORITY_QUEUE_H_
#define MACHINA_CORE_KERNELS_PRIORITY_QUEUE_H_

#include <deque>
#include <queue>
#include <vector>

#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.h"
#include "machina/core/kernels/typed_queue.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/platform/types.h"

namespace machina {

using PriorityTensorPair = std::pair<int64_t, Tensor>;

struct ComparePriorityTensorPair {
  // 0 is a higher priority than 1, -MAX_LONG is a higher priority
  // than MAX_LONG, etc.  Values coming in with a smaller
  // priority number will bubble to the front of the queue.
  bool operator()(const PriorityTensorPair& lhs,
                  const PriorityTensorPair& rhs) const {
    return lhs.first > rhs.first;
  }
};

class PriorityQueue
    : public TypedQueue<std::priority_queue<PriorityTensorPair,
                                            std::vector<PriorityTensorPair>,
                                            ComparePriorityTensorPair> > {
 public:
  PriorityQueue(int32_t capacity, const DataTypeVector& component_dtypes,
                const std::vector<TensorShape>& component_shapes,
                const string& name);

  absl::Status Initialize()
      override;  // Must be called before any other method.

  // Implementations of QueueInterface methods --------------------------------

  void TryEnqueue(const Tuple& tuple, OpKernelContext* ctx,
                  DoneCallback callback) override;
  void TryEnqueueMany(const Tuple& tuple, OpKernelContext* ctx,
                      DoneCallback callback) override;
  void TryDequeue(OpKernelContext* ctx, CallbackWithTuple callback) override;
  void TryDequeueMany(int num_elements, OpKernelContext* ctx,
                      bool allow_small_batch,
                      CallbackWithTuple callback) override;
  absl::Status MatchesNodeDef(const NodeDef& node_def) override;
  absl::Status MatchesPriorityNodeDefTypes(const NodeDef& node_def) const;
  absl::Status MatchesPriorityNodeDefShapes(const NodeDef& node_def) const;

  int32 size() const override {
    mutex_lock lock(mu_);
    return queues_[0].size();
  }

 private:
  ~PriorityQueue() override {}

  // Helper for dequeuing a single element from queues_.
  void DequeueLocked(OpKernelContext* ctx, Tuple* tuple)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  static absl::Status GetElementComponentFromBatch(const Tuple& tuple,
                                                   int index, int component,
                                                   OpKernelContext* ctx,
                                                   Tensor* out_element);

  PriorityQueue(const PriorityQueue&) = delete;
  void operator=(const PriorityQueue&) = delete;
};

}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_PRIORITY_QUEUE_H_
