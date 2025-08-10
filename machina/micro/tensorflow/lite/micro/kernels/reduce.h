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

#ifndef MACHINA_LITE_MICRO_KERNELS_REDUCE_H_
#define MACHINA_LITE_MICRO_KERNELS_REDUCE_H_

#include <cstdint>

#include "machina/lite/c/builtin_op_data.h"
#include "machina/lite/kernels/internal/types.h"
#include "machina/lite/micro/micro_common.h"

namespace tflite {

extern const int kMaxNumberOfAxis;
extern const int kMaxNumberOfReducedAxis;

struct OpDataReduce {
  int32_t multiplier;
  int shift;
  int temp_buffer_idx;
  int resolved_axis_idx;
  int input_zp;
  float input_scale;
  int output_zp;
  float output_scale;
  int num_output_elements;
  int num_axis;
};

TfLiteStatus PrepareMinMaxHelper(TfLiteContext* context, TfLiteNode* node,
                                 OpDataReduce* op_data);

TfLiteStatus PrepareMeanOrSumHelper(TfLiteContext* context, TfLiteNode* node,
                                    OpDataReduce* op_data);

TfLiteStatus EvalMaxHelper(TfLiteContext* context, TfLiteNode* node,
                           OpDataReduce* op_data);
TfLiteStatus EvalMinHelper(TfLiteContext* context, TfLiteNode* node,
                           OpDataReduce* op_data);
TfLiteStatus EvalMeanHelper(TfLiteContext* context, TfLiteNode* node,
                            OpDataReduce* op_data);
TfLiteStatus EvalSumHelper(TfLiteContext* context, TfLiteNode* node,
                           OpDataReduce* op_data);

TFLMRegistration Register_MEAN();
TFLMRegistration Register_REDUCE_MAX();
TFLMRegistration Register_REDUCE_MIN();
TFLMRegistration Register_SUM();

}  // namespace tflite

#endif  // MACHINA_LITE_MICRO_KERNELS_REDUCE_H_
