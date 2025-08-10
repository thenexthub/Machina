/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

// A graph pass that rewrites graph for propagating MKL layout as a tensor

#ifndef MACHINA_CORE_COMMON_RUNTIME_MKL_LAYOUT_PASS_H_
#define MACHINA_CORE_COMMON_RUNTIME_MKL_LAYOUT_PASS_H_

#ifdef INTEL_MKL

#include <sys/types.h>
#include <memory>
#include "machina/core/graph/graph.h"

namespace machina {
// Interface to invoke the pass for unit test
//
// Returns true if and only if 'g' is mutated.
extern bool RunMklLayoutRewritePass(std::unique_ptr<Graph>* g);
}  // namespace machina

#endif

#endif  // MACHINA_CORE_COMMON_RUNTIME_MKL_LAYOUT_PASS_H_
