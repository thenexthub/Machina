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

#ifndef MACHINA_DTENSOR_CC_SMALL_CONSTANT_OPTIMIZATION_H_
#define MACHINA_DTENSOR_CC_SMALL_CONSTANT_OPTIMIZATION_H_

#include <optional>

#include "machina/c/eager/c_api.h"
#include "machina/core/framework/node_def_builder.h"
#include "machina/dtensor/cc/tensor_layout.h"

namespace machina {
namespace dtensor {

// Attempt to convert small constant tensors into a constant NodeDef operation.
// This constant value will be available for constant propagation in DTensor and
// MLIR.

// This conversion is currently required for some DTensor operations. In
// particular, reductions require access to the axis argument at compilation
// time. While this is not strictly necessary, it greatly simplifies SPMD code
// generation and is generally available.
std::optional<NodeDef> ExtractSmallTensorValue(TFE_Context* context,
                                               TFE_TensorHandle* tensor,
                                               const Layout& layout,
                                               TF_Status* status);

// Returns true if the tensor proto of a and b are different.
bool NodeDefsHaveDifferentTensorProto(const NodeDef& a, const NodeDef& b);
}  // namespace dtensor
}  // namespace machina

#endif  // MACHINA_DTENSOR_CC_SMALL_CONSTANT_OPTIMIZATION_H_
