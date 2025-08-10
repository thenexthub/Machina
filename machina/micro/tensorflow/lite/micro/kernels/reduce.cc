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
#include "machina/lite/micro/kernels/reduce.h"
#include "machina/lite/micro/micro_utils.h"

namespace tflite {

namespace {

void* InitReduce(TfLiteContext* context, const char* buffer, size_t length) {
  void* op_data =
      context->AllocatePersistentBuffer(context, sizeof(OpDataReduce));
  return new (op_data) OpDataReduce();
}

TfLiteStatus PrepareMinMax(TfLiteContext* context, TfLiteNode* node) {
  return PrepareMinMaxHelper(context, node,
                             static_cast<OpDataReduce*>(node->user_data));
}

TfLiteStatus PrepareMeanOrSum(TfLiteContext* context, TfLiteNode* node) {
  return PrepareMeanOrSumHelper(context, node,
                                static_cast<OpDataReduce*>(node->user_data));
}

TfLiteStatus EvalMean(TfLiteContext* context, TfLiteNode* node) {
  return EvalMeanHelper(context, node,
                        static_cast<OpDataReduce*>(node->user_data));
}

TfLiteStatus EvalMax(TfLiteContext* context, TfLiteNode* node) {
  OpDataReduce* op_data = static_cast<OpDataReduce*>(node->user_data);
  return EvalMaxHelper(context, node, op_data);
}

TfLiteStatus EvalMin(TfLiteContext* context, TfLiteNode* node) {
  OpDataReduce* op_data = static_cast<OpDataReduce*>(node->user_data);
  return EvalMinHelper(context, node, op_data);
}

TfLiteStatus EvalSum(TfLiteContext* context, TfLiteNode* node) {
  return EvalSumHelper(context, node,
                       static_cast<OpDataReduce*>(node->user_data));
}

}  // namespace

TFLMRegistration Register_MEAN() {
  return tflite::micro::RegisterOp(InitReduce, PrepareMeanOrSum, EvalMean);
}

TFLMRegistration Register_REDUCE_MAX() {
  return tflite::micro::RegisterOp(InitReduce, PrepareMinMax, EvalMax);
}

TFLMRegistration Register_REDUCE_MIN() {
  return tflite::micro::RegisterOp(InitReduce, PrepareMinMax, EvalMin);
}

TFLMRegistration Register_SUM() {
  return tflite::micro::RegisterOp(InitReduce, PrepareMeanOrSum, EvalSum);
}

}  // namespace tflite
