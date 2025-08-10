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

#include "machina/lite/delegates/gpu/common/transformations/make_fully_connected.h"

#include <any>
#include <memory>
#include <string>
#include <vector>

#include "absl/types/any.h"
#include "machina/lite/delegates/gpu/common/model.h"
#include "machina/lite/delegates/gpu/common/model_transformer.h"
#include "machina/lite/delegates/gpu/common/operations.h"
#include "machina/lite/delegates/gpu/common/shape.h"
#include "machina/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace {

bool IsConvEquivalentToFullyConnected(const Convolution2DAttributes& attr) {
  return attr.weights.shape.w == 1 &&           //
         attr.weights.shape.h == 1 &&           //
         attr.strides == HW(1, 1) &&            //
         attr.dilations == HW(1, 1) &&          //
         attr.padding.prepended == HW(0, 0) &&  //
         attr.padding.appended == HW(0, 0);
}

class MakeFullyConnectedFromConvolution : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final {
    if (node->operation.type != ToString(OperationType::CONVOLUTION_2D)) {
      return {TransformStatus::SKIPPED, ""};
    }
    auto inputs = graph->FindInputs(node->id);
    if (inputs.size() != 1) {
      return {TransformStatus::SKIPPED, ""};
    }

    const auto& input_shape = inputs[0]->tensor.shape;
    if (input_shape.w != 1 || input_shape.h != 1) {
      return {TransformStatus::SKIPPED, ""};
    }

    const auto& conv_attr = std::any_cast<const Convolution2DAttributes&>(
        node->operation.attributes);
    if (!IsConvEquivalentToFullyConnected(conv_attr)) {
      return {TransformStatus::SKIPPED, ""};
    }

    FullyConnectedAttributes fc_attr;
    fc_attr.weights = conv_attr.weights;
    fc_attr.bias = conv_attr.bias;

    node->operation.attributes = fc_attr;
    node->operation.type = ToString(OperationType::FULLY_CONNECTED);
    return {TransformStatus::APPLIED,
            "Replaced convolution with fully connected."};
  }
};

}  // namespace

std::unique_ptr<NodeTransformation> NewMakeFullyConnectedFromConvolution() {
  return std::make_unique<MakeFullyConnectedFromConvolution>();
}

}  // namespace gpu
}  // namespace tflite
