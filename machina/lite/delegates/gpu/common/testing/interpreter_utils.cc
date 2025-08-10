/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include "machina/lite/delegates/gpu/common/testing/interpreter_utils.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "machina/lite/core/api/op_resolver.h"
#include "machina/lite/core/interpreter_builder.h"
#include "machina/lite/core/kernels/register.h"
#include "machina/lite/delegates/gpu/common/status.h"
#include "machina/lite/delegates/gpu/common/tensor.h"
#include "machina/lite/interpreter.h"
#include "machina/lite/schema/schema_generated.h"

namespace tflite {
namespace gpu {
namespace testing {

absl::Status InterpreterInvokeWithOpResolver(
    const ::tflite::Model* model, TfLiteDelegate* delegate,
    const OpResolver& op_resolver, const std::vector<TensorFloat32>& inputs,
    std::vector<TensorFloat32>* outputs) {
  auto interpreter = std::make_unique<Interpreter>();
  if (InterpreterBuilder(model, op_resolver)(&interpreter) != kTfLiteOk) {
    return absl::InternalError("Unable to create TfLite InterpreterBuilder");
  }
  if (delegate && interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
    return absl::InternalError(
        "Unable to modify TfLite graph with the delegate");
  }
  interpreter->SetNumThreads(1);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    return absl::InternalError("Unable to allocate TfLite tensors");
  }
  for (int i = 0; i < inputs.size(); ++i) {
    if (interpreter->tensor(interpreter->inputs()[i])->type != kTfLiteFloat32) {
      return absl::InternalError("input data_type is not float32");
    }
    float* tflite_data =
        interpreter->typed_tensor<float>(interpreter->inputs()[i]);
    if (inputs[i].data.size() * sizeof(float) >
        interpreter->tensor(interpreter->inputs()[i])->bytes) {
      return absl::InternalError("too big input data");
    }
    std::memcpy(tflite_data, inputs[i].data.data(),
                inputs[i].data.size() * sizeof(float));
  }
  if (interpreter->Invoke() != kTfLiteOk) {
    return absl::InternalError("Unable to invoke TfLite interpreter");
  }
  if (!outputs || !outputs->empty()) {
    return absl::InternalError("Invalid outputs pointer");
  }
  outputs->reserve(interpreter->outputs().size());
  for (auto t : interpreter->outputs()) {
    const TfLiteTensor* out_tensor = interpreter->tensor(t);
    TensorFloat32 bhwc;
    bhwc.id = t;
    // TODO(impjdi) Relax this condition to arbitrary batch size.
    if (out_tensor->dims->data[0] != 1) {
      return absl::InternalError("Batch dimension is expected to be 1");
    }
    bhwc.shape.b = out_tensor->dims->data[0];
    switch (out_tensor->dims->size) {
      case 2:
        bhwc.shape.h = 1;
        bhwc.shape.w = 1;
        bhwc.shape.c = out_tensor->dims->data[1];
        break;
      case 3:
        bhwc.shape.h = 1;
        bhwc.shape.w = out_tensor->dims->data[1];
        bhwc.shape.c = out_tensor->dims->data[2];
        break;
      case 4:
        bhwc.shape.h = out_tensor->dims->data[1];
        bhwc.shape.w = out_tensor->dims->data[2];
        bhwc.shape.c = out_tensor->dims->data[3];
        break;
      default:
        return absl::InternalError("Unsupported dimensions size " +
                                   std::to_string(out_tensor->dims->size));
    }
    bhwc.data = std::vector<float>(
        out_tensor->data.f,
        out_tensor->data.f + out_tensor->bytes / sizeof(float));
    outputs->push_back(bhwc);
  }
  return absl::OkStatus();
}

absl::Status InterpreterInvoke(const ::tflite::Model* model,
                               TfLiteDelegate* delegate,
                               const std::vector<TensorFloat32>& inputs,
                               std::vector<TensorFloat32>* outputs) {
  ops::builtin::BuiltinOpResolver builtin_op_resolver;
  return InterpreterInvokeWithOpResolver(model, delegate, builtin_op_resolver,
                                         inputs, outputs);
}

}  // namespace testing
}  // namespace gpu
}  // namespace tflite
