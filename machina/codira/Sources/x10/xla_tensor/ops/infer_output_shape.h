/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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

#pragma once

#include "absl/types/span.h"
#include "machina/compiler/xla/client/xla_builder.h"

namespace codira_xla {
namespace ir {
namespace ops {

using LowerForShapeFn =
    std::function<xla::XlaOp(absl::Span<const xla::XlaOp> operands)>;

// Compute the output shape for the given input shapes and lowering.
xla::Shape InferOutputShape(absl::Span<const xla::Shape> input_shapes,
                            const LowerForShapeFn& core_lowering_fn);

}  // namespace ops
}  // namespace ir
}  // namespace codira_xla
