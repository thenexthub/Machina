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

#include "machina/lite/delegates/gpu/common/flops_util.h"

#include "machina/lite/delegates/gpu/common/shape.h"

namespace tflite {
namespace gpu {

uint64_t GetConvolutionFlops(const BHWC& dst_shape, const OHWI& weights_shape) {
  uint64_t dst_elements = dst_shape.b * dst_shape.h * dst_shape.w * dst_shape.c;
  // 2 flops per operation( s = a * b + s);
  return dst_elements * weights_shape.i * weights_shape.w * weights_shape.h * 2;
}

uint64_t GetConvolutionWinograd4x4To6x6Flops(const BHWC& dst_shape,
                                             const OHWI& weights_shape) {
  return GetConvolutionFlops(dst_shape, weights_shape) / 4u;
}

uint64_t GetConvolutionTransposedFlops(const BHWC& src_shape,
                                       const OHWI& weights_shape) {
  uint64_t elements = src_shape.b * src_shape.h * src_shape.w * weights_shape.o;
  // 2 flops per operation( s = a * b + s);
  return elements * weights_shape.i * weights_shape.w * weights_shape.h * 2;
}

uint64_t GetDepthwiseConvolutionFlops(const BHWC& dst_shape,
                                      const OHWI& weights_shape) {
  uint64_t dst_elements = dst_shape.b * dst_shape.h * dst_shape.w * dst_shape.c;
  // 2 flops per operation( s = a * b + s);
  return dst_elements * weights_shape.w * weights_shape.h * 2;
}

uint64_t GetFullyConnectedFlops(const BHWC& dst_shape,
                                const OHWI& weights_shape) {
  uint64_t dst_elements = dst_shape.b * dst_shape.h * dst_shape.w * dst_shape.c;
  // 2 flops per operation( s = a * b + s);
  return dst_elements * weights_shape.i * 2;
}

}  // namespace gpu
}  // namespace tflite
