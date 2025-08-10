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
#ifndef MACHINA_LITE_TOOLS_OPTIMIZE_QUANTIZATION_WRAPPER_UTILS_H_
#define MACHINA_LITE_TOOLS_OPTIMIZE_QUANTIZATION_WRAPPER_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "machina/lite/core/api/error_reporter.h"
#include "machina/lite/core/model.h"
#include "machina/lite/schema/schema_generated.h"

namespace tflite {
namespace optimize {

// Load a tflite model from path.
TfLiteStatus LoadModel(const string& path, ModelT* model);

// Going through the model and add intermediates tensors if the ops have any.
// Returns early if the model has already intermediate tensors. This is to
// support cases where a model is initialized multiple times.
TfLiteStatus AddIntermediateTensorsToFusedOp(
    flatbuffers::FlatBufferBuilder* builder, ModelT* model);

// Write model to a given location.
bool WriteFile(const std::string& out_file, const uint8_t* bytes,
               size_t num_bytes);

}  // namespace optimize
}  // namespace tflite

#endif  // MACHINA_LITE_TOOLS_OPTIMIZE_QUANTIZATION_WRAPPER_UTILS_H_
