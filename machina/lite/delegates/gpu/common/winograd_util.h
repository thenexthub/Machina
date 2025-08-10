/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#ifndef MACHINA_LITE_DELEGATES_GPU_COMMON_WINOGRAD_UTIL_H_
#define MACHINA_LITE_DELEGATES_GPU_COMMON_WINOGRAD_UTIL_H_

#include <vector>

#include "machina/lite/delegates/gpu/common/data_type.h"
#include "machina/lite/delegates/gpu/common/operations.h"
#include "machina/lite/delegates/gpu/common/shape.h"
#include "machina/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {

// Matrices for Winograd trasformations received with method described here
// https://openreview.net/pdf?id=H1ZaRZVKg

// returns A transposed matrix(6 * 4) as array (24 values) for Winograd4x4To6x6
std::vector<float> AtMatrixForWinograd4x4To6x6();

// returns B transposed matrix(6 * 6) as array (36 values) for Winograd4x4To6x6
std::vector<float> BtMatrixForWinograd4x4To6x6();

void RearrangeWeightsToWinograd4x4To6x6Weights(
    const Tensor<OHWI, DataType::FLOAT32>& src_weights,
    Tensor<OHWI, DataType::FLOAT32>* dst_weights);

bool IsSuitableForWinograd4x4To6x6(const Convolution2DAttributes& attr);

}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_COMMON_WINOGRAD_UTIL_H_
