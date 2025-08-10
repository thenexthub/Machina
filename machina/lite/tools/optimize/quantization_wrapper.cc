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
#include "machina/lite/tools/optimize/quantization_wrapper.h"

#include <string>

#include "machina/lite/tools/optimize/quantization_wrapper_utils.h"
#include "machina/lite/tools/optimize/quantize_model.h"

namespace tflite {
namespace optimize {

bool CreateModelForCalibration(const std::string& input_path,
                               const std::string& output_path) {
  ModelT model;
  if (LoadModel(input_path, &model) != kTfLiteOk) {
    return false;
  }
  flatbuffers::FlatBufferBuilder builder;
  if (AddIntermediateTensorsToFusedOp(&builder, &model) != kTfLiteOk) {
    return false;
  }
  return WriteFile(output_path, builder.GetBufferPointer(), builder.GetSize());
}

bool CreateQuantizedModel(const std::string& path) {
  ModelT model;
  if (LoadModel(path, &model) != kTfLiteOk) {
    return false;
  }
  flatbuffers::FlatBufferBuilder builder;
  tflite::StderrReporter error_reporter;
  if (tflite::optimize::QuantizeModel(
          &builder, &model, tflite::TensorType_FLOAT32,
          tflite::TensorType_FLOAT32,
          // TODO(b/159351372): Pass required activation type if needed
          tflite::TensorType_INT8, &error_reporter) != kTfLiteOk) {
    return false;
  }
  return WriteFile(path, builder.GetBufferPointer(), builder.GetSize());
}

}  // namespace optimize
}  // namespace tflite
