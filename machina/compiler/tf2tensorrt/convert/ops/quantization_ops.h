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
#ifndef MACHINA_COMPILER_TF2TENSORRT_CONVERT_OPS_QUANTIZATION_OPS_H_
#define MACHINA_COMPILER_TF2TENSORRT_CONVERT_OPS_QUANTIZATION_OPS_H_
#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "machina/core/graph/graph.h"
#include "machina/core/lib/core/status.h"

namespace machina {
namespace tensorrt {
namespace convert {

constexpr std::array<const char*, 4> kQuantizationOpNames = {
    "QuantizeAndDequantizeV2",
    "QuantizeAndDequantizeV3",
    "FakeQuantWithMinMaxVars",
    "FakeQuantWithMinMaxArgs",
};

// Operations with supported conversion to Q/DQ ops in TensorRT explicit
// precision mode.
constexpr std::array<const char*, 1> kExplicitQuantizationOpNames = {
    "QuantizeAndDequantizeV2",
};

// Contains two scaling factors for quantization and dequantization
// respectively. A shift factor is omitted as TensorRT only supports symmetric
// quantization.
template <typename T, size_t N>
struct QuantizationScales {
  std::array<T, N> quantize_scale;
  std::array<T, N> dequantize_scale;
};

// In TensorRT 7 and 8, only uniform tensor scaling is supported for
// activations.
using UniformQuantizationScales = QuantizationScales<float, 1>;

// Per-channel scaling is supported for weights in TensorRT version >= 8.0.
template <size_t ChannelDimSize>
using PerChannelQuantizationScales = QuantizationScales<float, ChannelDimSize>;

template <typename T, size_t N>
std::ostream& operator<<(std::ostream& os,
                         const QuantizationScales<T, N>& scales) {
  os << absl::StrFormat("QuantizationScales[quantize={%s},dequantize={%s}]",
                        absl::StrJoin(scales.quantize_scale, ","),
                        absl::StrJoin(scales.dequantize_scale, ","));
  return os;
}

// Returns true if the Tensorflow node is a quantize and dequantize operation.
bool IsQuantizeAndDequantizeOp(const Node*);

}  // namespace convert
}  // namespace tensorrt
}  // namespace machina

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

#endif  // MACHINA_COMPILER_TF2TENSORRT_CONVERT_OPS_QUANTIZATION_OPS_H_
