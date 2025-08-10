/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#include "machina/lite/micro/kernels/softmax.h"

#include "machina/lite/c/builtin_op_data.h"
#include "machina/lite/c/common.h"
#include "machina/lite/kernels/internal/common.h"
#include "machina/lite/kernels/internal/quantization_util.h"
#include "machina/lite/kernels/internal/reference/softmax.h"
#include "machina/lite/kernels/internal/tensor_ctypes.h"
#include "machina/lite/kernels/kernel_util.h"
#include "machina/lite/kernels/op_macros.h"
#include "machina/lite/micro/kernels/kernel_util.h"
#include "machina/lite/micro/micro_log.h"

namespace tflite {
namespace {

void SoftmaxQuantized(const TfLiteEvalTensor* input, TfLiteEvalTensor* output,
                      const SoftmaxParams& op_data) {
  if (input->type == kTfLiteInt8) {
    if (output->type == kTfLiteInt16) {
      tflite::reference_ops::Softmax(
          op_data, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int16_t>(output));
    } else {
      tflite::reference_ops::Softmax(
          op_data, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
    }
  } else {
    tflite::reference_ops::SoftmaxInt16(
        op_data, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int16_t>(input),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int16_t>(output));
  }
}

TfLiteStatus SoftmaxEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  TFLITE_DCHECK(node->user_data != nullptr);
  SoftmaxParams op_data = *static_cast<SoftmaxParams*>(node->user_data);

  switch (input->type) {
    case kTfLiteFloat32: {
      tflite::reference_ops::Softmax(
          op_data, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output));
      return kTfLiteOk;
    }
    case kTfLiteInt8:
    case kTfLiteInt16: {
      SoftmaxQuantized(input, output, op_data);
      return kTfLiteOk;
    }
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
}
}  // namespace

TFLMRegistration Register_SOFTMAX() {
  return tflite::micro::RegisterOp(SoftmaxInit, SoftmaxPrepare, SoftmaxEval);
}

}  // namespace tflite
