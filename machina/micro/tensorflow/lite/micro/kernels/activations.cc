/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, August 10, 2025.
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

#include "machina/lite/micro/kernels/activations.h"

#include "machina/lite/c/builtin_op_data.h"
#include "machina/lite/c/common.h"
#include "machina/lite/kernels/internal/common.h"
#include "machina/lite/kernels/internal/quantization_util.h"
#include "machina/lite/kernels/internal/tensor_ctypes.h"
#include "machina/lite/kernels/internal/types.h"
#include "machina/lite/kernels/kernel_util.h"
#include "machina/lite/kernels/op_macros.h"
#include "machina/lite/micro/kernels/kernel_util.h"
#include "machina/lite/micro/micro_log.h"
#include "machina/lite/micro/micro_utils.h"

namespace tflite {
namespace {

void* ReluInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(ReluOpData));
}

TfLiteStatus ReluEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const ReluOpData& data = *(static_cast<const ReluOpData*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kActivationsInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kActivationsOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32: {
      ReluFloat(tflite::micro::GetTensorShape(input),
                tflite::micro::GetTensorData<float>(input),
                tflite::micro::GetTensorShape(output),
                tflite::micro::GetTensorData<float>(output));

      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      tflite::ReluQuantized<int8_t>(
          data, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorData<int8_t>(output));
      return kTfLiteOk;
    }
    case kTfLiteInt16: {
      tflite::ReluQuantized<int16_t>(
          data, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int16_t>(input),
          tflite::micro::GetTensorData<int16_t>(output));
      return kTfLiteOk;
    }
    default: {
      MicroPrintf("Only float32/int8/int16 is supported currently, got %s",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
}

void* Relu6Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(Relu6OpData));
}

TfLiteStatus Relu6Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);

  const Relu6OpData& data = *(static_cast<const Relu6OpData*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kActivationsInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kActivationsOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32: {
      Relu6Float(tflite::micro::GetTensorShape(input),
                 tflite::micro::GetTensorData<float>(input),
                 tflite::micro::GetTensorShape(output),
                 tflite::micro::GetTensorData<float>(output));

      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      Relu6Quantized<int8_t>(data.zero, data.six,
                             tflite::micro::GetTensorShape(input),
                             tflite::micro::GetTensorData<int8_t>(input),
                             tflite::micro::GetTensorShape(output),
                             tflite::micro::GetTensorData<int8_t>(output));
      return kTfLiteOk;
    }
    case kTfLiteInt16: {
      Relu6Quantized<int16_t>(data.zero, data.six,
                              tflite::micro::GetTensorShape(input),
                              tflite::micro::GetTensorData<int16_t>(input),
                              tflite::micro::GetTensorShape(output),
                              tflite::micro::GetTensorData<int16_t>(output));
      return kTfLiteOk;
    }
    default: {
      MicroPrintf("Only float32/int8/int16 is supported currently, got %s",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
}

}  // namespace

TFLMRegistration Register_RELU() {
  return tflite::micro::RegisterOp(ReluInit, ReluPrepare, ReluEval);
}

TFLMRegistration Register_RELU6() {
  return tflite::micro::RegisterOp(Relu6Init, Relu6Prepare, Relu6Eval);
}

}  // namespace tflite
