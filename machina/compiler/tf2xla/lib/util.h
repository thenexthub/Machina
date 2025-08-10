/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#ifndef MACHINA_COMPILER_TF2MACHINA_XLALIB_UTIL_H_
#define MACHINA_COMPILER_TF2MACHINA_XLALIB_UTIL_H_

#include <cstdint>

#include "absl/types/span.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/xla/hlo/builder/xla_computation.h"
#include "machina/xla/xla_data.pb.h"
#include "machina/core/platform/statusor.h"

namespace machina {

// Returns a floating point scalar constant of 'type' with 'value'.
// If 'type' is complex, returns a real value with zero imaginary component.
xla::XlaOp FloatLiteral(xla::XlaBuilder* builder, xla::PrimitiveType type,
                        double value);

// Makes a 1D tensor [0, ..., x, y] from two tensors x and y with zeros
// prepended until the array is length n_dims.
xla::XlaOp PrependZerosInMajorDims(xla::XlaOp x,
                                   absl::Span<const xla::XlaOp> starts);

// Returns a integer scalar constant of 'type' with 'value'.
// If 'type' is complex, returns a real value with zero imaginary component.
xla::XlaOp IntegerLiteral(xla::XlaBuilder* builder, xla::PrimitiveType type,
                          int64_t value);

}  // namespace machina

#endif  // MACHINA_COMPILER_TF2MACHINA_XLALIB_UTIL_H_
