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

#include "machina/lite/micro/mock_micro_graph.h"

#include "machina/lite/micro/test_helpers.h"

namespace tflite {

MockMicroGraph::MockMicroGraph(SingleArenaBufferAllocator* allocator)
    : allocator_(allocator), init_count_(0), prepare_count_(0), free_count_(0) {
  memset(invoke_counts_, 0, sizeof(invoke_counts_));
  mock_tensor_ =
      reinterpret_cast<TfLiteEvalTensor*>(allocator_->AllocatePersistentBuffer(
          sizeof(TfLiteEvalTensor), alignof(TfLiteEvalTensor)));
  int* dims_array = reinterpret_cast<int*>(
      allocator_->AllocatePersistentBuffer(3 * sizeof(int), alignof(int)));
  float* data_array = reinterpret_cast<float*>(
      allocator_->AllocatePersistentBuffer(2 * sizeof(float), alignof(float)));
  int dims[] = {2, 1, 2};
  memcpy(dims_array, dims, 3 * sizeof(int));
  mock_tensor_->dims = testing::IntArrayFromInts(dims_array);
  mock_tensor_->data.f = data_array;
  mock_tensor_->type = kTfLiteFloat32;
}

TfLiteStatus MockMicroGraph::InvokeSubgraph(int subgraph_idx) {
  invoke_counts_[subgraph_idx]++;
  return kTfLiteOk;
}

size_t MockMicroGraph::NumSubgraphInputs(int subgraph_idx) { return 1; }

TfLiteEvalTensor* MockMicroGraph::GetSubgraphInput(int subgraph_idx,
                                                   int tensor_idx) {
  return mock_tensor_;
}

size_t MockMicroGraph::NumSubgraphOutputs(int subgraph_idx) { return 1; }

TfLiteEvalTensor* MockMicroGraph::GetSubgraphOutput(int subgraph_idx,
                                                    int tensor_idx) {
  return mock_tensor_;
}

int MockMicroGraph::NumSubgraphs() { return kMaxSubgraphs; }

MicroResourceVariables* MockMicroGraph::GetResourceVariables() {
  return nullptr;
}

}  // namespace tflite
