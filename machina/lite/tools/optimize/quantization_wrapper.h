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
#ifndef MACHINA_LITE_TOOLS_OPTIMIZE_QUANTIZATION_WRAPPER_H_
#define MACHINA_LITE_TOOLS_OPTIMIZE_QUANTIZATION_WRAPPER_H_

#include <string>

namespace tflite {
namespace optimize {

// Makes an copy of the model at input_path and writes it to output_path, adding
// tensors to the model needed for calibration.
// Returns true if it is successful.
// Example: a/b/c.tflite becomes a/b/c.calibrated.tflite and has
// intermediate tensors added according to operator properties.
bool CreateModelForCalibration(const std::string& input_path,
                               const std::string& output_path);

// Quantize a model in place. This function is only to be called after calling
// CreateModelForCalibration and running calibration over data.
// Returns true if it is successful.
bool CreateQuantizedModel(const std::string& path);

}  // namespace optimize
}  // namespace tflite

#endif  // MACHINA_LITE_TOOLS_OPTIMIZE_QUANTIZATION_WRAPPER_H_
