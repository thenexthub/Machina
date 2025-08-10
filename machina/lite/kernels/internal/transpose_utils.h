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
#ifndef MACHINA_LITE_KERNELS_INTERNAL_TRANSPOSE_UTILS_H_
#define MACHINA_LITE_KERNELS_INTERNAL_TRANSPOSE_UTILS_H_

#include "machina/lite/kernels/internal/types.h"

namespace tflite {
namespace transpose_utils {

// IsTranspose2DApplicable returns true if the given perm can be lowered to a
// 2D transpose op. If possible, it copies the lowered dimension counts to the
// given dim0 and dim1 pointers.
bool IsTranspose2DApplicable(const TransposeParams& params,
                             const RuntimeShape& input_shape, int* dim0,
                             int* dim1);

// RemoveOneSizeDimensions removes one size dimensions in the given input/output
// shapes and adjusts the parameter values for transpose op.
void RemoveOneSizeDimensions(RuntimeShape* input_shape,
                             RuntimeShape* output_shape,
                             TransposeParams* params);

// Flatten finds the dimensions that can be flatten, shrinks the given shapes
// and the given perm parameter to reflect the non-flatten dimensions, and
// returns the total size of the non-flatten dimensions.
//
// E.g, in perm [0, 1, 3, 2] case, the first two dimensions can be flatten and
// it returns |Dim Size(2)| x |Dim Size(3)|.
size_t Flatten(const RuntimeShape& input_shape,
               const RuntimeShape& output_shape, const TransposeParams& params,
               RuntimeShape* non_flatten_input_shape,
               RuntimeShape* non_flatten_output_shape,
               TransposeParams* non_flatten_params);

}  // namespace transpose_utils

}  // namespace tflite

#endif  // MACHINA_LITE_KERNELS_INTERNAL_TRANSPOSE_UTILS_H_
