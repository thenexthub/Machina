/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#ifndef MACHINA_TSL_PLATFORM_CONTEXT_H_
#define MACHINA_TSL_PLATFORM_CONTEXT_H_

#include "tsl/platform/platform.h"

namespace tsl {

enum class ContextKind {
  // Initial state with default (empty) values.
  kDefault,
  // Initial state inherited from the creating or scheduling thread.
  kThread,
};

// Context is a container for request-specific information that should be passed
// to threads that perform related work. The default constructor should capture
// all relevant context.
class Context;

// Scoped object that sets the current thread's context until the object is
// destroyed.
class WithContext;

}  // namespace tsl

#if defined(PLATFORM_GOOGLE)
#include "machina/xla/tsl/platform/google/context.h"  // IWYU pragma: export
#else
#include "machina/xla/tsl/platform/default/context.h"  // IWYU pragma: export
#endif

#endif  // MACHINA_TSL_PLATFORM_CONTEXT_H_
