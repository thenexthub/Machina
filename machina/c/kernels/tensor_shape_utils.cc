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

#include "machina/c/kernels/tensor_shape_utils.h"

#include <string>

#include "machina/c/tf_tensor.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/strcat.h"

namespace machina {

std::string ShapeDebugString(TF_Tensor* tensor) {
  // A TF_Tensor cannot have an unknown rank.
  CHECK_GE(TF_NumDims(tensor), 0);
  machina::string s = "[";
  for (int i = 0; i < TF_NumDims(tensor); ++i) {
    if (i > 0) machina::strings::StrAppend(&s, ",");
    int64_t dim = TF_Dim(tensor, i);
    // A TF_Tensor cannot have an unknown dimension.
    CHECK_GE(dim, 0);
    machina::strings::StrAppend(&s, dim);
  }
  machina::strings::StrAppend(&s, "]");
  return s;
}
}  // namespace machina
