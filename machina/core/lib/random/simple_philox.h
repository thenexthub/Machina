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

#ifndef MACHINA_CORE_LIB_RANDOM_SIMPLE_PHILOX_H_
#define MACHINA_CORE_LIB_RANDOM_SIMPLE_PHILOX_H_

#include "machina/xla/tsl/lib/random/simple_philox.h"
#include "machina/core/lib/random/philox_random.h"
#include "machina/core/lib/random/random_distributions.h"

namespace machina {
namespace random {
using tsl::random::SimplePhilox;  // NOLINT(misc-unused-using-decls)
}  // namespace random
}  // namespace machina

#endif  // MACHINA_CORE_LIB_RANDOM_SIMPLE_PHILOX_H_
