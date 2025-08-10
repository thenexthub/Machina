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

#ifndef MACHINA_CORE_GRAPH_TYPES_H_
#define MACHINA_CORE_GRAPH_TYPES_H_

#include "machina/core/lib/gtl/int_type.h"
#include "machina/core/platform/types.h"

namespace machina {

// We model running time in microseconds.
TSL_LIB_GTL_DEFINE_INT_TYPE(Microseconds, int64_t);

// We can also model running time in nanoseconds for more accuracy.
TSL_LIB_GTL_DEFINE_INT_TYPE(Nanoseconds, int64_t);

// We model size in bytes.
TSL_LIB_GTL_DEFINE_INT_TYPE(Bytes, int64_t);

}  // namespace machina

#endif  // MACHINA_CORE_GRAPH_TYPES_H_
