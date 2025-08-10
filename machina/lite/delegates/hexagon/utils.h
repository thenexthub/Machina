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
#ifndef MACHINA_LITE_DELEGATES_HEXAGON_UTILS_H_
#define MACHINA_LITE_DELEGATES_HEXAGON_UTILS_H_

#include "machina/lite/core/c/common.h"

namespace tflite {

// Interpretes data from 'dims' as a 4D shape {batch, height, width, depth} and
// populates the corresponding values. If dims->size < 4, the shape is prefixed
// with 1s.
// For example, dims {2, 3} is interpreted as: {1, 1, 2, 3}.
// Returns kTfLiteError if dims->size > 4, kTfLiteOk otherwise.
TfLiteStatus Get4DShape(unsigned int* batch_size, unsigned int* height_size,
                        unsigned int* width_size, unsigned int* depth_size,
                        TfLiteIntArray* dims);

// Returns true if provided node is supported by Hexagon NNLib in the current
// context.
bool IsNodeSupportedByHexagon(const TfLiteRegistration* registration,
                              const TfLiteNode* node, TfLiteContext* context);

}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_HEXAGON_UTILS_H_
