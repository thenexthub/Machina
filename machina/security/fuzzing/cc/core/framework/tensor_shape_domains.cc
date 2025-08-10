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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "fuzztest/fuzztest.h"
#include "absl/status/statusor.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/core/framework/tensor_shape.h"

namespace machina::fuzzing {
namespace {

using ::fuzztest::Domain;
using ::fuzztest::Filter;
using ::fuzztest::InRange;
using ::fuzztest::Map;
using ::fuzztest::VectorOf;

Domain<absl::StatusOr<TensorShape>> AnyStatusOrTensorShape(
    size_t max_rank, int64_t dim_lower_bound, int64_t dim_upper_bound) {
  return Map(
      [](std::vector<int64_t> v) -> absl::StatusOr<TensorShape> {
        TensorShape out;
        TF_RETURN_IF_ERROR(TensorShape::BuildTensorShape(v, &out));
        return out;
      },
      VectorOf(InRange(dim_lower_bound, dim_upper_bound))
          .WithMaxSize(max_rank));
}

}  // namespace

Domain<TensorShape> AnyValidTensorShape(
    size_t max_rank = std::numeric_limits<size_t>::max(),
    int64_t dim_lower_bound = std::numeric_limits<int64_t>::min(),
    int64_t dim_upper_bound = std::numeric_limits<int64_t>::max()) {
  return Map([](absl::StatusOr<TensorShape> t) { return *t; },
             Filter([](auto t_inner) { return t_inner.status().ok(); },
                    AnyStatusOrTensorShape(max_rank, dim_lower_bound,
                                           dim_upper_bound)));
}

}  // namespace machina::fuzzing
