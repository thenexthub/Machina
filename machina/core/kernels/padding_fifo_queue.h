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

#ifndef MACHINA_CORE_KERNELS_PADDING_FIFO_QUEUE_H_
#define MACHINA_CORE_KERNELS_PADDING_FIFO_QUEUE_H_

#include <deque>
#include <vector>

#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/partial_tensor_shape.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.h"
#include "machina/core/kernels/fifo_queue.h"
#include "machina/core/kernels/typed_queue.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/platform/types.h"

namespace machina {

class PaddingFIFOQueue : public FIFOQueue {
 public:
  PaddingFIFOQueue(int32_t capacity, const DataTypeVector& component_dtypes,
                   const std::vector<PartialTensorShape>& component_shapes,
                   const string& name);

  absl::Status Initialize() override;

  // Implementations of QueueInterface methods --------------------------------

  void TryDequeueMany(int num_elements, OpKernelContext* ctx,
                      bool allow_small_batch,
                      CallbackWithTuple callback) override;
  absl::Status MatchesNodeDef(const NodeDef& node_def) override;

 protected:
  absl::Status ValidateManyTuple(const Tuple& tuple) override;
  absl::Status ValidateTuple(const Tuple& tuple) override;
  absl::Status CompatibleNodeDefShapes(const NodeDef& node_def) const;

  // Convert a list of PartialTensorShape to a list of
  // TensorShape.
  // Any unknown dimension sizes are converted to 0.
  // REQUIRED: All the input shapes have well defined rank.
  static std::vector<TensorShape> ConvertShapesPartialDimensionsToZero(
      absl::Span<const PartialTensorShape> partial_shapes);

  // Sets the values in the given element to zero.
  static absl::Status SetElementZero(Tensor* element);

  // Copies element into the index^th slice (in the first dimension)
  // of parent.  Allows for the parent's slice to have a larger size
  // than the element, and copies the element into the upper left hand
  // corner of the slice.
  static absl::Status CopyElementToLargerSlice(const Tensor& element,
                                               Tensor* parent, int index);

  std::vector<PartialTensorShape> partial_shapes_;

 private:
  ~PaddingFIFOQueue() override {}

  static absl::Status GetElementComponent(const PaddingFIFOQueue::Tuple& tuple,
                                          int component, OpKernelContext* ctx,
                                          Tensor* out_tensor);

  static absl::Status IsSameSizeExceptZerosInFirst(const TensorShape& first,
                                                   const TensorShape& second);

  PaddingFIFOQueue(const PaddingFIFOQueue&) = delete;
  void operator=(const PaddingFIFOQueue&) = delete;
};

}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_PADDING_FIFO_QUEUE_H_
