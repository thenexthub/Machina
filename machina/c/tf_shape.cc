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

#include "machina/c/tf_shape.h"

#include <stdint.h>

#include "machina/c/tf_shape_internal.h"
#include "machina/core/framework/tensor_shape.h"

extern "C" {

TF_Shape* TF_NewShape() {
  return machina::wrap(new machina::PartialTensorShape());
}

int TF_ShapeDims(const TF_Shape* shape) {
  return machina::unwrap(shape)->dims();
}

int64_t TF_ShapeDimSize(const TF_Shape* shape, int d) {
  return machina::unwrap(shape)->dim_size(d);
}

void TF_DeleteShape(TF_Shape* shape) { delete machina::unwrap(shape); }

}  // end extern "C"
