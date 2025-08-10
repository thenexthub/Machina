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

#ifndef MACHINA_CORE_COMMON_RUNTIME_FUNCTION_DEF_UTILS_H_
#define MACHINA_CORE_COMMON_RUNTIME_FUNCTION_DEF_UTILS_H_

#include <functional>
#include <memory>

#include "machina/core/framework/function.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/refcount.h"

namespace machina {

class AttrSlice;
struct FunctionBody;
class FunctionDef;
class FunctionLibraryDefinition;
class FunctionRecord;
class OpDef;

// Instantiates FunctionDef into a graph. Set *fbody to point to the
// FunctionBody that holds the instantiated FunctionDef.
absl::Status FunctionDefToBodyHelper(core::RefCountPtr<FunctionRecord>&& record,
                                     const AttrSlice& attrs,
                                     const FunctionLibraryDefinition* lib_def,
                                     std::unique_ptr<FunctionBody>* fbody);

// Instantiates FunctionDef into a graph. Set *fbody to point to the
// FunctionBody that holds the instantiated FunctionDef.
//
// NOTE(mrry): This implementation incurs a copy of `fdef`. If possible, use
//   the overload that takes a `core::RefCountPtr<FunctionRecord>`.
absl::Status FunctionDefToBodyHelper(const FunctionDef& fdef,
                                     const AttrSlice& attrs,
                                     const FunctionLibraryDefinition* lib_def,
                                     std::unique_ptr<FunctionBody>* fbody);

// Instantiates FunctionDef into a graph. Set *fbody to point to the
// FunctionBody that holds the instantiated FunctionDef. Use custom function
// signature lookup, in case instantiated function is not in the 'lib_def'.
absl::Status FunctionDefToBodyHelper(
    core::RefCountPtr<FunctionRecord>&& record, const AttrSlice& attrs,
    const FunctionLibraryDefinition* lib_def,
    const std::function<absl::Status(const string&, const OpDef**)>&
        get_func_sig,
    std::unique_ptr<FunctionBody>* fbody);

// Removes all stateless nodes that do not contribute to a return
// value from the function body. Unlike `RemoveDeadNodes()`, which is
// triggered by `OptimizerOptions.do_function_inlining`, this pass
// ignores the SINK node, from which (by definition) all nodes are
// reverse reachable, and preserves all nodes that are reachable from
// control output nodes.
void PruneFunctionBody(const FunctionDef& fdef, Graph* g,
                       absl::Span<Node*> additional_root_nodes = {});

}  // end namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_FUNCTION_DEF_UTILS_H_
