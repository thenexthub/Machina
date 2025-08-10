/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
#include "machina/lite/kernels/internal/optimized/optimized_ops.h"
#include "machina/lite/kernels/internal/tensor.h"
#include "machina/lite/kernels/internal/tensor_ctypes.h"
#include "machina/lite/kernels/kernel_util.h"
#include "machina/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace ceil {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
  output->type = input->type;
  TfLiteIntArray* output_size = TfLiteIntArrayCopy(input->dims);
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  if (input->type != kTfLiteFloat32) {
    TF_LITE_UNSUPPORTED_TYPE(context, input->type, "Ceil");
  }

  optimized_ops::Ceil(GetTensorShape(input), GetTensorData<float>(input),
                      GetTensorShape(output), GetTensorData<float>(output));

  return kTfLiteOk;
}
}  // namespace ceil

TfLiteRegistration* Register_CEIL() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr, ceil::Prepare, ceil::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
