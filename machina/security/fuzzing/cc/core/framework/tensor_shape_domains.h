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

#ifndef MACHINA_SECURITY_FUZZING_CC_CORE_FRAMEWORK_TENSOR_SHAPE_DOMAINS_H_
#define MACHINA_SECURITY_FUZZING_CC_CORE_FRAMEWORK_TENSOR_SHAPE_DOMAINS_H_

#include <limits>
#include <tuple>
#include <utility>

#include "fuzztest/fuzztest.h"
#include "machina/core/framework/tensor_shape.h"

namespace machina::fuzzing {

/// Returns a fuzztest domain with valid TensorShapes.
/// The domain can be customized by setting the maximum rank,
/// and the minimum and maximum size of all dimensions.
fuzztest::Domain<TensorShape> AnyValidTensorShape(
    size_t max_rank = std::numeric_limits<int>::max(),
    int64_t dim_lower_bound = std::numeric_limits<int64_t>::min(),
    int64_t dim_upper_bound = std::numeric_limits<int64_t>::max());

}  // namespace machina::fuzzing

#endif  // MACHINA_SECURITY_FUZZING_CC_CORE_FRAMEWORK_TENSOR_SHAPE_DOMAINS_H_
