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
#ifndef MACHINA_LITE_EXPERIMENTAL_DELEGATES_COREML_COREML_DELEGATE_KERNEL_H_
#define MACHINA_LITE_EXPERIMENTAL_DELEGATES_COREML_COREML_DELEGATE_KERNEL_H_

#include "machina/lite/core/c/common.h"
#include "machina/lite/delegates/coreml/builders/op_builder.h"
#import "machina/lite/delegates/coreml/coreml_executor.h"

namespace tflite {
namespace delegates {
namespace coreml {

// Represents a subgraph in TFLite that will be delegated to CoreML.
// It is abstracted as a single kernel node in the main TFLite graph and
// implements Init/Prepare/Invoke as TFLite kernel nodes.
class CoreMlDelegateKernel {
 public:
  explicit CoreMlDelegateKernel(int coreml_version)
      : coreml_version_(coreml_version) {}
  // Initialize the delegated graph and add required nodes.
  TfLiteStatus Init(TfLiteContext* context, const TfLiteDelegateParams* params);

  // Any preparation work needed for the delegated graph.
  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node);

  // Allocates delegated tensordefs for graph I/O & execute it.
  TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node);

  ~CoreMlDelegateKernel();

 private:
  // Builds the ML Model protocol buffer
  TfLiteStatus BuildModel(TfLiteContext* context,
                          const TfLiteDelegateParams* params);

  // Adds the output tensors to the model generated.
  void AddOutputTensors(const TfLiteIntArray* output_tensors,
                        TfLiteContext* context);

  // Adds the input tensors to the model generated.
  void AddInputTensors(const TfLiteIntArray* output_tensors,
                       TfLiteContext* context);

  std::unique_ptr<delegates::coreml::GraphBuilder> builder_;
  std::unique_ptr<CoreML::Specification::Model> model_;
  ::CoreMlExecutor* executor_;
  int coreml_version_;

  std::vector<int> input_tensor_ids_;
  std::vector<TensorData> inputs_;
  std::vector<TensorData> outputs_;
};

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite

#endif  // MACHINA_LITE_EXPERIMENTAL_DELEGATES_COREML_COREML_DELEGATE_KERNEL_H_
