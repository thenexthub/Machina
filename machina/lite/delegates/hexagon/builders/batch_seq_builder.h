/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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
#ifndef MACHINA_LITE_DELEGATES_HEXAGON_BUILDERS_BATCH_SEQ_BUILDER_H_
#define MACHINA_LITE_DELEGATES_HEXAGON_BUILDERS_BATCH_SEQ_BUILDER_H_

#include "machina/lite/delegates/hexagon/builders/op_builder.h"

namespace tflite {
namespace delegates {
namespace hexagon {

class BatchSeqBuilder : public OpBuilder {
 public:
  explicit BatchSeqBuilder(GraphBuilder* graph_builder, int op_type)
      : OpBuilder(graph_builder, op_type) {}

  TfLiteStatus PopulateSubGraph(const TfLiteIntArray* inputs,
                                const TfLiteIntArray* outputs,
                                TfLiteContext* context) override;

  TfLiteStatus RegisterOutputs(const TfLiteIntArray* outputs,
                               TfLiteContext* context) override {
    // BatchSeqConfig doesn't have any outputs.
    return kTfLiteOk;
  }

  void SetMaxSizeForBatch(int max_size_for_batch) {
    max_size_for_batch_ = max_size_for_batch;
  }

  void SetInputBatchDimensions(TfLiteIntArray* input_batch_dimensions) {
    input_batch_dims_ = input_batch_dimensions;
  }

  void SetOutputBatchDimensions(TfLiteIntArray* output_batch_dimensions) {
    output_batch_dims_ = output_batch_dimensions;
  }

 private:
  // Maximum size for the batch dimension in a single run.
  // The graph can have input with larger batch, internally
  // multiple runs will happen each won't have more than 'max_size_for_batch_'
  // in batch dimension.
  int max_size_for_batch_ = 1;
  // Input dimension for each input in the graph.
  // Input with fixed batch should have -1.
  TfLiteIntArray* input_batch_dims_;
  // Output dimension for each output in the graph.
  // Output with fixed batch should have -1.
  TfLiteIntArray* output_batch_dims_;
};

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_HEXAGON_BUILDERS_BATCH_SEQ_BUILDER_H_
