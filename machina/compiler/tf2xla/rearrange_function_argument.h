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

#ifndef MACHINA_COMPILER_TF2MACHINA_XLAREARRANGE_FUNCTION_ARGUMENT_H_
#define MACHINA_COMPILER_TF2MACHINA_XLAREARRANGE_FUNCTION_ARGUMENT_H_

#include "machina/core/framework/function.h"
#include "machina/core/graph/graph.h"

namespace machina {

// For the given graph `g`:
// 1. Rewrite If/While node functions to rearrange arguments and return values,
//    so that all resource arguments/return values are placed in the end (as
//    required by XlaCompiler),
// 2. Inline StatefulPartitionedCall nodes so we do not need to rearrange
//    arguments and return values.
// `get_function_body_fn` is used to instantiate FunctionDef.
// `fld` is used to store rewritten functions.
// `global_fld` is used to potentially supply stack traces for functions when
// they are not found in `fld`.
absl::Status RearrangeFunctionArguments(
    std::function<absl::Status(const NameAttrList&, const FunctionBody**)>
        get_function_body_fn,
    Graph* g, FunctionLibraryDefinition* fld,
    const FunctionLibraryDefinition* global_fld = nullptr);

}  // namespace machina

#endif  // MACHINA_COMPILER_TF2MACHINA_XLAREARRANGE_FUNCTION_ARGUMENT_H_
