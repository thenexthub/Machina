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
#include "machina/lite/micro/kernels/logical.h"

#include "machina/lite/c/common.h"
#include "machina/lite/kernels/internal/reference/binary_function.h"
#include "machina/lite/kernels/internal/tensor_ctypes.h"
#include "machina/lite/kernels/op_macros.h"
#include "machina/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {

TfLiteStatus LogicalOrEval(TfLiteContext* context, TfLiteNode* node) {
  return LogicalImpl(context, node, LogicalOr);
}

TfLiteStatus LogicalAndEval(TfLiteContext* context, TfLiteNode* node) {
  return LogicalImpl(context, node, LogicalAnd);
}

}  // namespace

TFLMRegistration Register_LOGICAL_OR() {
  return tflite::micro::RegisterOp(nullptr, nullptr, LogicalOrEval);
}

TFLMRegistration Register_LOGICAL_AND() {
  return tflite::micro::RegisterOp(nullptr, nullptr, LogicalAndEval);
}

}  // namespace tflite
