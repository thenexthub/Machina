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

#include "machina/lite/kernels/internal/reference/reduce.h"

#include "machina/lite/c/builtin_op_data.h"
#include "machina/lite/c/common.h"
#include "machina/lite/kernels/internal/quantization_util.h"
#include "machina/lite/kernels/internal/reference/integer_ops/mean.h"
#include "machina/lite/kernels/internal/tensor_ctypes.h"
#include "machina/lite/kernels/internal/types.h"
#include "machina/lite/kernels/kernel_util.h"
#include "machina/lite/micro/kernels/kernel_util.h"
#include "machina/lite/micro/kernels/xtensa/xtensa.h"
#include "machina/lite/micro/kernels/xtensa/xtensa_reduce.h"
#include "machina/lite/micro/micro_utils.h"

namespace tflite {

void* XtensaInitReduce(TfLiteContext* context, const char* buffer,
                       size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  void* data =
      context->AllocatePersistentBuffer(context, sizeof(XtensaReduceOpData));

#if defined(VISION_P6)
  if (InitXtensaContext() != 0) {
    return nullptr;
  }
#endif
  return data;
}

TfLiteStatus XtensaPrepareMax(TfLiteContext* context, TfLiteNode* node) {
  OpDataReduce* op_data =
      &(static_cast<XtensaReduceOpData*>(node->user_data)->reference_op_data);
  TF_LITE_ENSURE_OK(context, PrepareMinMaxHelper(context, node, op_data));
#if defined(VISION_P6)
  TF_LITE_ENSURE_OK(context, ReducePrepareVision(context, node));
#endif  // VISION_P6
  return kTfLiteOk;
}

TfLiteStatus XtensaPrepareMin(TfLiteContext* context, TfLiteNode* node) {
  OpDataReduce* op_data =
      &(static_cast<XtensaReduceOpData*>(node->user_data)->reference_op_data);
  TF_LITE_ENSURE_OK(context, PrepareMinMaxHelper(context, node, op_data));
  // P6 FLK library does not support REDUCE_MIN
  return kTfLiteOk;
}

TfLiteStatus XtensaPrepareMeanOrSum(TfLiteContext* context, TfLiteNode* node) {
  OpDataReduce* op_data =
      &(static_cast<XtensaReduceOpData*>(node->user_data)->reference_op_data);
  return PrepareMeanOrSumHelper(context, node, op_data);
}

TfLiteStatus XtensaEvalMean(TfLiteContext* context, TfLiteNode* node) {
  OpDataReduce* op_data =
      &(static_cast<XtensaReduceOpData*>(node->user_data)->reference_op_data);
  return EvalMeanHelper(context, node, op_data);
}

TfLiteStatus XtensaEvalMax(TfLiteContext* context, TfLiteNode* node) {
  XtensaReduceOpData* op_data_xtensa =
      static_cast<XtensaReduceOpData*>(node->user_data);
  OpDataReduce* op_data = &(op_data_xtensa->reference_op_data);

#if defined(VISION_P6)
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  switch (input->type) {
    case kTfLiteInt8: {
      TF_LITE_ENSURE_EQ(context, static_cast<double>(op_data->input_scale),
                        static_cast<double>(op_data->output_scale));
      TF_LITE_ENSURE_EQ(context, op_data->input_zp, op_data->output_zp);
      ReduceEvalVision(*op_data_xtensa, input, output);
      break;
    }
    default: {
      // Use the reference EvalMax for all other cases.
      return EvalMaxHelper(context, node, op_data);
    }
  }
  return kTfLiteOk;
#else
  return EvalMaxHelper(context, node, op_data);
#endif
}

TfLiteStatus XtensaEvalMin(TfLiteContext* context, TfLiteNode* node) {
  XtensaReduceOpData* op_data_xtensa =
      static_cast<XtensaReduceOpData*>(node->user_data);
  OpDataReduce* op_data = &(op_data_xtensa->reference_op_data);
  // P6 FLK library does not support REDUCE_MIN
  return EvalMinHelper(context, node, op_data);
}

TfLiteStatus XtensaEvalSum(TfLiteContext* context, TfLiteNode* node) {
  OpDataReduce* op_data =
      &(static_cast<XtensaReduceOpData*>(node->user_data)->reference_op_data);
  return EvalSumHelper(context, node, op_data);
}

TFLMRegistration Register_MEAN() {
  return tflite::micro::RegisterOp(XtensaInitReduce, XtensaPrepareMeanOrSum,
                                   XtensaEvalMean);
}

TFLMRegistration Register_REDUCE_MAX() {
  return tflite::micro::RegisterOp(XtensaInitReduce, XtensaPrepareMax,
                                   XtensaEvalMax);
}

TFLMRegistration Register_REDUCE_MIN() {
  return tflite::micro::RegisterOp(XtensaInitReduce, XtensaPrepareMin,
                                   XtensaEvalMin);
}

TFLMRegistration Register_SUM() {
  return tflite::micro::RegisterOp(XtensaInitReduce, XtensaPrepareMeanOrSum,
                                   XtensaEvalSum);
}

}  // namespace tflite
