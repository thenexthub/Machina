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

#if defined(VISION_P6)

#include <cstdint>

#include "machina/lite/c/builtin_op_data.h"
#include "machina/lite/c/common.h"
#include "machina/lite/kernels/internal/common.h"
#include "machina/lite/kernels/internal/reference/reduce.h"
#include "machina/lite/kernels/internal/tensor_ctypes.h"
#include "machina/lite/kernels/kernel_util.h"
#include "machina/lite/micro/kernels/kernel_util.h"
#include "machina/lite/micro/kernels/reshape.h"
#include "machina/lite/micro/kernels/xtensa/xtensa.h"
#include "machina/lite/micro/kernels/xtensa/xtensa_reshape.h"

namespace tflite {

inline void OperandDims4D(uint32_t* dims, TfLiteTensor* opnd) {
  for (int i = NumDimensions(opnd) - 1, j = 0; i >= 0; i--, j++) {
    dims[j] = SizeOfDimension(opnd, i);
  }
  return;
}

TfLiteStatus ReshapePrepareVision(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  XtensaReshapeData* data =
      reinterpret_cast<XtensaReshapeData*>(node->user_data);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kReshapeInputTensor);

  uint32_t inputRank = NumDimensions(input);
  uint32_t inputDims[4] = {1, 1, 1, 1};
  OperandDims4D(inputDims, input);
  uint32_t context_size = 0;
  uint32_t status = xiReshapeGetMemReqd_Context(&context_size);
  TFLITE_DCHECK(status == 0);
  if (context_size) {
    void* context_data =
        context->AllocatePersistentBuffer(context, context_size);
    if (context_data == nullptr) {
      return kTfLiteError;
    }
    data->p_context = reinterpret_cast<uint8_t*>(context_data);
    data->context_size = context_size;
  }

  status = xiReshapeSetContext(data->p_context, data->context_size, inputDims,
                               inputRank);

  if (status) {
    return kTfLiteError;
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  return kTfLiteOk;
}
TfLiteStatus ReshapeEvalVision(const XtensaReshapeData& data,
                               const TfLiteEvalTensor* input,
                               TfLiteEvalTensor* output) {
  const uint32_t input_size = NumElements(input->dims);
  const uint32_t output_size = NumElements(output->dims);

  xiReshape(data.p_context, data.context_size,
            const_cast<int8_t*>(tflite::micro::GetTensorData<int8_t>(input)),
            input_size, tflite::micro::GetTensorData<int8_t>(output),
            output_size);
  return kTfLiteOk;
}
}  // namespace tflite
#endif  // defined(VISION_P6)
