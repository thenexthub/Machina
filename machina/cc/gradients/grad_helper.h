/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#ifndef MACHINA_CC_GRADIENTS_GRAD_HELPER_H_
#define MACHINA_CC_GRADIENTS_GRAD_HELPER_H_

#include "machina/cc/framework/ops.h"
#include "machina/cc/framework/scope.h"

namespace machina {

// Helper function for reduction ops.
//
// input_shape: 1-D Tensor, the shape of the Tensor being reduced.
// axes: 1-D Tensor, the reduction axes.
//   Note that the reduction indices are in the range
//   -rank(input_shape), rank(input_shape)
// returns a 1-D Tensor, the output shape as if keep_dims were set to True.
Output ReducedShapeHelper(const Scope& scope, const Output& input_shape,
                          const Output& reduction_axes);

}  // namespace machina

#endif  // MACHINA_CC_GRADIENTS_GRAD_HELPER_H_
