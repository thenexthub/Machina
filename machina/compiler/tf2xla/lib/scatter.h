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

#ifndef MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_LIB_SCATTER_H_
#define MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_LIB_SCATTER_H_

#include <functional>

#include "absl/status/statusor.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/xla/hlo/builder/xla_computation.h"
#include "machina/core/platform/statusor.h"

namespace machina {

// Builds an XLA computation that performs a scatter operation on `buffer`,
// returning an updated buffer.
// For each i0, i1, ..., sets
// buffer[indices[i0, i1, ...], ...] := updates[i0, i1, ...]
//
// If `indices_are_vectors` is false, then each index in indices is a scalar,
// and the shape of `indices` must be a prefix of the shape of updates.
// Otherwise, `indices_are_vectors`, then indices are multidimensional and the
// minor dimension of `indices` represents a vector of indices.
//
// If `updates` is a scalar, then it will be broadcasted into the expected shape
// of updates.
//
// If any part of the update region is out-of-bounds, the corresponding update
// is discarded.
//
// If a `combiner` is provided, updates are combined with the existing values in
// the buffer using the combiner function. Otherwise, the updates replace the
// existing values. The order of updates is implementation-defined.
absl::StatusOr<xla::XlaOp> XlaScatter(
    xla::XlaOp buffer, xla::XlaOp updates, xla::XlaOp indices,
    bool indices_are_vectors, bool indices_are_sorted,
    const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp, xla::XlaBuilder*)>&
        combiner,
    xla::XlaBuilder* builder);

}  // namespace machina

#endif  // MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_LIB_SCATTER_H_
