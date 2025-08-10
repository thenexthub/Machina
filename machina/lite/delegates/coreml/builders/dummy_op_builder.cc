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
#include "machina/lite/delegates/coreml/builders/dummy_op_builder.h"

#include <string>

#include "mlmodel/format/NeuralNetwork.pb.h"
#include "machina/lite/c/c_api_types.h"
#include "machina/lite/c/common.h"
#include "machina/lite/delegates/coreml/builders/op_factory.h"

namespace tflite {
namespace delegates {
namespace coreml {

CoreML::Specification::NeuralNetworkLayer* DummyOpBuilder::Build() {
  return nullptr;
}

const std::string& DummyOpBuilder::DebugName() {
  SetDebugName("DummyOpBuilder", node_id_);
  return debug_name_;
}

TfLiteStatus DummyOpBuilder::PopulateSubgraph(TfLiteContext* context) {
  return kTfLiteOk;
}

OpBuilder* CreateDummyOpBuilder(GraphBuilder* graph_builder) {
  return new DummyOpBuilder(graph_builder);
}

TfLiteStatus DummyOpBuilder::RegisterInputs(const TfLiteIntArray* inputs,
                                            TfLiteContext* context) {
  return kTfLiteOk;
}

TfLiteStatus DummyOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                             TfLiteContext* context) {
  return kTfLiteOk;
}

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
