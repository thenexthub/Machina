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
#include "machina/lite/core/c/common.h"
#include "machina/lite/core/subgraph.h"
#include "machina/lite/experimental/resource/lookup_interfaces.h"
#include "machina/lite/kernels/internal/tensor_ctypes.h"
#include "machina/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {

namespace hashtable {

constexpr int kInputResourceIdTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus PrepareHashtableSize(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input_resource_id_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputResourceIdTensor,
                                          &input_resource_id_tensor));
  TF_LITE_ENSURE_EQ(context, input_resource_id_tensor->type, kTfLiteResource);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_resource_id_tensor), 1);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(input_resource_id_tensor, 0), 1);

  TfLiteTensor* output_tensor;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kOutputTensor, &output_tensor));
  TF_LITE_ENSURE_EQ(context, output_tensor->type, kTfLiteInt64);
  TfLiteIntArray* outputSize = TfLiteIntArrayCreate(1);
  outputSize->data[0] = 1;
  return context->ResizeTensor(context, output_tensor, outputSize);
}

TfLiteStatus EvalHashtableSize(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input_resource_id_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputResourceIdTensor,
                                          &input_resource_id_tensor));
  int resource_id = input_resource_id_tensor->data.i32[0];

  TfLiteTensor* output_tensor;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kOutputTensor, &output_tensor));
  auto* output_data = GetTensorData<std::int64_t>(output_tensor);

  Subgraph* subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto& resources = subgraph->resources();
  auto* lookup = resource::GetHashtableResource(&resources, resource_id);
  TF_LITE_ENSURE(context, lookup != nullptr);

  output_data[0] = lookup->Size();
  return kTfLiteOk;
}

}  // namespace hashtable

TfLiteRegistration* Register_HASHTABLE_SIZE() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 hashtable::PrepareHashtableSize,
                                 hashtable::EvalHashtableSize};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
