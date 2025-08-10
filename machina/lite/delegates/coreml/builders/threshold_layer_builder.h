/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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
#ifndef MACHINA_LITE_DELEGATES_COREML_BUILDERS_THRESHOLD_LAYER_BUILDER_H_
#define MACHINA_LITE_DELEGATES_COREML_BUILDERS_THRESHOLD_LAYER_BUILDER_H_

#include <string>

#include "mlmodel/format/NeuralNetwork.pb.h"
#include "machina/lite/c/c_api_types.h"
#include "machina/lite/c/common.h"
#include "machina/lite/delegates/coreml/builders/op_builder.h"

namespace tflite {
namespace delegates {
namespace coreml {

// Layer that provides threshold operation. Depending on scale, this can be used
// as max (scale > 0) or min (scale < 0), in combination with another negative
// linear activation layer) operation.
// TODO(karimnosseir): Generalize to other unary operators.
class ThresholdLayerBuilder : public OpBuilder {
 public:
  explicit ThresholdLayerBuilder(GraphBuilder* graph_builder)
      : OpBuilder(graph_builder) {}

  const std::string& DebugName() override;

  CoreML::Specification::NeuralNetworkLayer* Build() override;

  void SetAlpha(float alpha) { alpha_ = alpha; }

  void SetScale(float scale) { scale_ = scale; }

  TfLiteStatus RegisterInputs(const TfLiteIntArray* inputs,
                              TfLiteContext* context) override;

  TfLiteStatus RegisterOutputs(const TfLiteIntArray* outputs,
                               TfLiteContext* context) override;

 private:
  float alpha_ = 0.0f;
  float scale_ = 1.0f;
};

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_COREML_BUILDERS_THRESHOLD_LAYER_BUILDER_H_
