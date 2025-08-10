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

#include <stdint.h>

#include "machina/lite/core/c/common.h"
#include "machina/lite/core/subgraph.h"
#include "machina/lite/experimental/resource/resource_variable.h"
#include "machina/lite/kernels/internal/tensor.h"
#include "machina/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace assign_variable {

constexpr int kInputVariableId = 0;
constexpr int kInputValue = 1;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 0);

  const TfLiteTensor* input_resource_id_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputVariableId,
                                          &input_resource_id_tensor));
  TF_LITE_ENSURE(context, (input_resource_id_tensor->type == kTfLiteResource ||
                           input_resource_id_tensor->type == kTfLiteInt32));
  TF_LITE_ENSURE_EQ(context, NumElements(input_resource_id_tensor), 1);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  Subgraph* subgraph = reinterpret_cast<Subgraph*>(context->impl_);

  const TfLiteTensor* input_resource_id_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputVariableId,
                                          &input_resource_id_tensor));
  const TfLiteTensor* input_value_tensor;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kInputValue, &input_value_tensor));

  int resource_id = input_resource_id_tensor->data.i32[0];
  auto& resources = subgraph->resources();
  resource::CreateResourceVariableIfNotAvailable(&resources, resource_id);
  auto* variable = resource::GetResourceVariable(&resources, resource_id);
  TF_LITE_ENSURE(context, variable != nullptr);
  variable->AssignFrom(input_value_tensor);

  return kTfLiteOk;
}

}  // namespace assign_variable

TfLiteRegistration* Register_ASSIGN_VARIABLE() {
  static TfLiteRegistration r = {nullptr, nullptr, assign_variable::Prepare,
                                 assign_variable::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
