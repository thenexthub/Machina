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

#ifndef MACHINA_CORE_KERNELS_STATEFUL_RANDOM_OPS_H_
#define MACHINA_CORE_KERNELS_STATEFUL_RANDOM_OPS_H_

#include "machina/core/framework/types.h"
#include "machina/core/lib/random/philox_random.h"

namespace machina {

// 'Variable' doesn't support uint32 or uint64 yet (due to reasons explained
// in b/111604096 and cl/171681867), so we use signed int here. We choose int64
// instead of int32 because `VarHandleOp` doesn't support int32 on GPU, and
// because of the "int32 problem".
using StateElementType = int64_t;
static constexpr DataType STATE_ELEMENT_DTYPE = DT_INT64;
static constexpr DataType ALGORITHM_DTYPE = STATE_ELEMENT_DTYPE;

using random::PhiloxRandom;

static constexpr int64_t PHILOX_MIN_STATE_SIZE =
    (PhiloxRandom::ResultType::kElementCount +
     PhiloxRandom::Key::kElementCount) /
    2;
static constexpr int64_t THREEFRY_MIN_STATE_SIZE = 2;

}  // end namespace machina

#endif  // MACHINA_CORE_KERNELS_STATEFUL_RANDOM_OPS_H_
