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

#ifndef MACHINA_LITE_DELEGATES_GPU_COMMON_QUANTIZATION_UTIL_H_
#define MACHINA_LITE_DELEGATES_GPU_COMMON_QUANTIZATION_UTIL_H_

#include <stdint.h>

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "machina/lite/core/c/common.h"
#include "machina/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {

// Dequantizes input tensors pre-inference, leaving float tensors intact.
// input_indices contains dequantized (fp32) outputs, that are used as
// inputs to GPU delegate.
// quant_conversion_map contains bidirectional mapping between dequantized
// tensor and its original quantized one.
absl::Status DequantizeInputs(
    TfLiteContext* context, const std::vector<uint32_t>& input_indices,
    const absl::flat_hash_map<int, int>& quant_conversion_map);

absl::Status DequantizeInputs(
    TfLiteContext* context, const std::vector<int64_t>& input_indices,
    const absl::flat_hash_map<int, int>& quant_conversion_map);

// Quantizes output tensors post-inference, leaving float tensors intact.
// output_indices contains (fp32) inputs to be quantized, which are outputs of
// GPU delegate.
// quant_conversion_map contains bidirectional mapping between dequantized
// tensor and its original quantized one.
absl::Status QuantizeOutputs(
    TfLiteContext* context, const std::vector<uint32_t>& output_indices,
    const absl::flat_hash_map<int, int>& quant_conversion_map);

absl::Status QuantizeOutputs(
    TfLiteContext* context, const std::vector<int64_t>& output_indices,
    const absl::flat_hash_map<int, int>& quant_conversion_map);
}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_COMMON_QUANTIZATION_UTIL_H_
