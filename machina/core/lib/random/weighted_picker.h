/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

// An abstraction to pick from one of N elements with a specified
// weight per element.
//
// The weight for a given element can be changed in O(lg N) time
// An element can be picked in O(lg N) time.
//
// Uses O(N) bytes of memory.
//
// Alternative: distribution-sampler.h allows O(1) time picking, but no weight
// adjustment after construction.

#ifndef MACHINA_CORE_LIB_RANDOM_WEIGHTED_PICKER_H_
#define MACHINA_CORE_LIB_RANDOM_WEIGHTED_PICKER_H_

#include <assert.h>

#include "machina/xla/tsl/lib/random/weighted_picker.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace random {
using tsl::random::WeightedPicker;  // NOLINT(misc-unused-using-decls)
}  // namespace random
}  // namespace machina

#endif  // MACHINA_CORE_LIB_RANDOM_WEIGHTED_PICKER_H_
