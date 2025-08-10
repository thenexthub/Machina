/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#include "machina/lite/kernels/internal/reference/maximum_minimum.h"

#include "machina/lite/c/builtin_op_data.h"
#include "machina/lite/c/common.h"
#include "machina/lite/kernels/internal/common.h"
#include "machina/lite/kernels/internal/quantization_util.h"
#include "machina/lite/kernels/internal/tensor_ctypes.h"
#include "machina/lite/kernels/kernel_util.h"
#include "machina/lite/kernels/op_macros.h"
#include "machina/lite/micro/kernels/kernel_util.h"
#include "machina/lite/micro/kernels/maximum_minimum.h"
#include "machina/lite/micro/micro_log.h"

namespace tflite {

namespace {

template <KernelType kernel_type, typename OpType>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);

  if (kernel_type == kReference) {
    switch (op_context.output->type) {
      case kTfLiteFloat32:
        TFLiteOperation<float, OpType>(context, node, op_context);
        break;
      case kTfLiteInt8:
        TFLiteOperation<int8_t, OpType>(context, node, op_context);
        break;
      case kTfLiteInt16:
        TFLiteOperation<int16_t, OpType>(context, node, op_context);
        break;
      case kTfLiteInt32:
        TFLiteOperation<int32_t, OpType>(context, node, op_context);
        break;
      case kTfLiteInt64:
        TFLiteOperation<int64_t, OpType>(context, node, op_context);
        break;
      default:
        MicroPrintf("Type %s (%d) is not supported by Maximum/Minimum.",
                    TfLiteTypeGetName(op_context.output->type),
                    op_context.output->type);
        return kTfLiteError;
    }
  } else {
    MicroPrintf("Kernel type not supported by Maximum/Minimum.");
    return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_MAXIMUM() {
  return tflite::micro::RegisterOp(nullptr, nullptr,
                                   Eval<kReference, MaximumOp>);
}

TFLMRegistration Register_MINIMUM() {
  return tflite::micro::RegisterOp(nullptr, nullptr,
                                   Eval<kReference, MinimumOp>);
}

}  // namespace tflite
