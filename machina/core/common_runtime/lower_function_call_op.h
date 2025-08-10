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

#ifndef MACHINA_CORE_COMMON_RUNTIME_LOWER_FUNCTION_CALL_OP_H_
#define MACHINA_CORE_COMMON_RUNTIME_LOWER_FUNCTION_CALL_OP_H_

#include "machina/core/lib/core/status.h"

namespace machina {

class FunctionLibraryDefinition;
class Graph;
class Node;

// Replaces function call node `n` with its function body. Uses
// InlineFunctionBody from `common_runtime/function.{h,cc}`. If function
// inlining is not possible or safe (see ValidateInlining), leaves the graph in
// unmodified state and returns OkStatus();
absl::Status RewriteFunctionCallNode(Node* n, Graph* g,
                                     const FunctionLibraryDefinition& flib_def,
                                     bool keep_caller_fetchable);

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_LOWER_FUNCTION_CALL_OP_H_
