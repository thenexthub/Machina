/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#ifndef MACHINA_LITE_MICRO_KERNELS_XTENSA_XTENSA_POOLING_H_
#define MACHINA_LITE_MICRO_KERNELS_XTENSA_XTENSA_POOLING_H_

#include <cstdint>

#include "machina/lite/c/builtin_op_data.h"
#include "machina/lite/c/common.h"
#include "machina/lite/micro/kernels/pooling.h"
namespace tflite {

struct XtensaOpDataPooling {
  OpDataPooling reference_op_data;

#if defined(VISION_P6)
  uint8_t* p_context;  // persistent lib context for this instance saved here
  uint32_t context_size;
#endif  // defined(VISION_P6)

#if defined(HIFI5)
  int scratch_tensor_index;
#endif  // defined(HIFI5)
};

#if defined(VISION_P6)

TfLiteStatus AvgPoolingPrepareVision(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus MaxPoolingPrepareVision(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus PoolEvalVision(TfLiteContext* context, TfLiteNode* node,
                            const TfLitePoolParams& params,
                            const XtensaOpDataPooling& data,
                            const TfLiteEvalTensor* input,
                            TfLiteEvalTensor* output);
#endif

#if defined(HIFI5)

TfLiteStatus AveragePrepareHifi(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus AverageEvalQuantizedHifi(TfLiteContext* context,
                                      const TfLiteNode* node,
                                      const TfLitePoolParams* params,
                                      const XtensaOpDataPooling* data,
                                      const TfLiteEvalTensor* input,
                                      TfLiteEvalTensor* output);

TfLiteStatus MaxPrepareHifi(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus MaxEvalQuantizedHifi(TfLiteContext* context, TfLiteNode* node,
                                  TfLitePoolParams* params,
                                  const XtensaOpDataPooling* data,
                                  const TfLiteEvalTensor* input,
                                  TfLiteEvalTensor* output);

#endif  // defined(HIFI5)

void* XtensaPoolingInit(TfLiteContext* context, const char* buffer,
                        size_t length);

}  // namespace tflite

#endif  // MACHINA_LITE_MICRO_KERNELS_XTENSA_XTENSA_POOLING_H_
