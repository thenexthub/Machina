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
#include "machina/lite/delegates/coreml/builders/threshold_layer_builder.h"

#include <memory>
#include <string>

#include "mlmodel/format/NeuralNetwork.pb.h"
#include "machina/lite/core/c/common.h"
#include "machina/lite/delegates/coreml/builders/op_builder.h"

namespace tflite {
namespace delegates {
namespace coreml {

const std::string& ThresholdLayerBuilder::DebugName() {
  if (debug_name_.empty()) SetDebugName("ThresholdLayerBuilder", node_id_);
  return debug_name_;
}

CoreML::Specification::NeuralNetworkLayer* ThresholdLayerBuilder::Build() {
  if (layer_ == nullptr) {
    layer_ = std::make_unique<CoreML::Specification::NeuralNetworkLayer>();
  }
  layer_->set_name(DebugName());
  layer_->mutable_unary()->set_alpha(alpha_);
  layer_->mutable_unary()->set_scale(scale_);
  layer_->mutable_unary()->set_type(
      CoreML::Specification::UnaryFunctionLayerParams::THRESHOLD);
  return layer_.release();
}

TfLiteStatus ThresholdLayerBuilder::RegisterInputs(const TfLiteIntArray* inputs,
                                                   TfLiteContext* context) {
  if (inputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Threshold: Wrong # of inputs!.");
    return kTfLiteError;
  }
  AddInput(inputs->data[0]);
  return kTfLiteOk;
}

TfLiteStatus ThresholdLayerBuilder::RegisterOutputs(
    const TfLiteIntArray* outputs, TfLiteContext* context) {
  if (outputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Threshold: Wrong # of outputs!.");
    return kTfLiteError;
  }
  graph_builder_->AddTensorWithID(outputs->data[0], GetOutput(context));
  return kTfLiteOk;
}

OpBuilder* CreateThresholdLayerBuilder(GraphBuilder* graph_builder) {
  return new ThresholdLayerBuilder(graph_builder);
}

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
