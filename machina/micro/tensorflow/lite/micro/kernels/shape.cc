/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

#include "machina/lite/c/builtin_op_data.h"
#include "machina/lite/c/common.h"
#include "machina/lite/kernels/internal/tensor_ctypes.h"
#include "machina/lite/kernels/kernel_util.h"
#include "machina/lite/kernels/op_macros.h"
#include "machina/lite/micro/kernels/kernel_util.h"
#include "machina/lite/micro/memory_helpers.h"
#include "machina/lite/micro/micro_log.h"
#include "machina/lite/micro/micro_utils.h"

namespace tflite {

namespace {
constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

void ExtractShape(const TfLiteEvalTensor* input, int32_t* output_data) {
  for (int i = 0; i < input->dims->size; ++i) {
    output_data[i] = input->dims->data[i];
  }
}

TfLiteStatus ShapePrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  return kTfLiteOk;
}

TfLiteStatus ShapeEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  if (output->type != kTfLiteInt32) {
    MicroPrintf("Output type %s (%d) not supported.",
                TfLiteTypeGetName(output->type), output->type);
    return kTfLiteError;
  } else {
    ExtractShape(input, tflite::micro::GetTensorData<int32_t>(output));
  }

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_SHAPE() {
  return tflite::micro::RegisterOp(nullptr, ShapePrepare, ShapeEval);
}

}  // namespace tflite
