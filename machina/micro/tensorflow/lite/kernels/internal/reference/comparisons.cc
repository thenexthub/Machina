/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#include "machina/lite/kernels/internal/reference/comparisons.h"

#include "machina/lite/kernels/internal/common.h"
#include "machina/lite/kernels/internal/compatibility.h"
#include "machina/lite/kernels/internal/runtime_shape.h"

namespace tflite {
namespace reference_ops {

BroadcastComparison4DSlowCommon BroadcastComparison4DSlowPreprocess(
    const RuntimeShape& unextended_input1_shape,
    const RuntimeShape& unextended_input2_shape,
    const RuntimeShape& unextended_output_shape) {
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);
  return {RuntimeShape::ExtendedShape(4, unextended_output_shape), desc1,
          desc2};
}

}  // namespace reference_ops
}  // namespace tflite
