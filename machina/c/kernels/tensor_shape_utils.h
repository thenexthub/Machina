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

// This file contains shape utilities to be used by kernels and is not part of
// the C API. As such, it is subject to change at any time.

#ifndef MACHINA_C_KERNELS_TENSOR_SHAPE_UTILS_H_
#define MACHINA_C_KERNELS_TENSOR_SHAPE_UTILS_H_

#include <string>

#include "machina/c/tf_tensor.h"

namespace machina {

// The following are utils for the shape of a TF_Tensor type.
// These functions may later be subsumed by the methods for a
// TF_TensorShape type.

// Returns a string representation of the TF_Tensor shape.
std::string ShapeDebugString(TF_Tensor* tensor);

}  // namespace machina

#endif  // MACHINA_C_KERNELS_TENSOR_SHAPE_UTILS_H_
