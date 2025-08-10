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
#ifndef MACHINA_LITE_DELEGATES_HEXAGON_BUILDERS_CAST_BUILDER_H_
#define MACHINA_LITE_DELEGATES_HEXAGON_BUILDERS_CAST_BUILDER_H_

#include <vector>

#include "machina/lite/delegates/hexagon/builders/op_builder.h"

namespace tflite {
namespace delegates {
namespace hexagon {

// This builder is used to cast int8 input or output tensors to & from uint8
// respectively. No TFLite op converts to this.
// NOTE: There are no explicit tests for this, but is required for all int8 unit
// tests.
class CastOpBuilder : public OpBuilder {
 public:
  explicit CastOpBuilder(GraphBuilder* graph_builder, int op_type)
      : OpBuilder(graph_builder, op_type) {}
  // inputs & outputs should contain the *same* (one) TFLite tensor-id, since
  // tensors are cast in-place. The tensor will point to a different Hexagon
  // TensorID after this runs.
  TfLiteStatus PopulateSubGraph(const TfLiteIntArray* inputs,
                                const TfLiteIntArray* outputs,
                                TfLiteContext* context) override;

  TfLiteStatus RegisterOutputs(const TfLiteIntArray* outputs,
                               TfLiteContext* context) override;

  ~CastOpBuilder() override;

 private:
  TensorID node_output_;
};

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_HEXAGON_BUILDERS_CAST_BUILDER_H_
