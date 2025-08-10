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

#ifndef MACHINA_LITE_MICRO_MOCK_MICRO_GRAPH_H_
#define MACHINA_LITE_MICRO_MOCK_MICRO_GRAPH_H_

#include "machina/lite/c/common.h"
#include "machina/lite/micro/micro_allocator.h"
#include "machina/lite/micro/micro_graph.h"
#include "machina/lite/schema/schema_generated.h"

namespace tflite {

// MockMicroGraph stubs out all MicroGraph methods used during invoke. A count
// of the number of calls to invoke for each subgraph is maintained for
// validation of control flow operators.
class MockMicroGraph : public MicroGraph {
 public:
  explicit MockMicroGraph(SingleArenaBufferAllocator* allocator);
  TfLiteStatus InvokeSubgraph(int subgraph_idx) override;
  size_t NumSubgraphInputs(int subgraph_idx) override;
  TfLiteEvalTensor* GetSubgraphInput(int subgraph_idx, int tensor_idx) override;
  size_t NumSubgraphOutputs(int subgraph_idx) override;
  TfLiteEvalTensor* GetSubgraphOutput(int subgraph_idx,
                                      int tensor_idx) override;
  int NumSubgraphs() override;
  MicroResourceVariables* GetResourceVariables() override;
  int get_init_count() const { return init_count_; }
  int get_prepare_count() const { return prepare_count_; }
  int get_free_count() const { return free_count_; }
  int get_invoke_count(int subgraph_idx) const {
    return invoke_counts_[subgraph_idx];
  }

 private:
  static constexpr int kMaxSubgraphs = 10;
  SingleArenaBufferAllocator* allocator_;
  TfLiteEvalTensor* mock_tensor_;
  int init_count_;
  int prepare_count_;
  int free_count_;
  int invoke_counts_[kMaxSubgraphs];
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // MACHINA_LITE_MICRO_MOCK_MICRO_GRAPH_H_
