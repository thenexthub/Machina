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

#ifndef MACHINA_CORE_COMMON_RUNTIME_FUNCTION_BODY_H_
#define MACHINA_CORE_COMMON_RUNTIME_FUNCTION_BODY_H_

#include "absl/status/status.h"
#include "machina/core/framework/allocator.h"
#include "machina/core/framework/function.h"
#include "machina/core/framework/types.h"
#include "machina/core/lib/gtl/inlined_vector.h"
#include "machina/core/platform/refcount.h"

namespace machina {

class FunctionRecord;
class Graph;
class Node;

// FunctionLibraryRuntime::GetFunctionBody returns a description of an
// instantiated function that is represented as a Graph with arg/ret
// nodes annotated.
struct FunctionBody {
  core::RefCountPtr<FunctionRecord> record;
  Graph* graph = nullptr;  // owned.
  DataTypeVector arg_types;
  DataTypeVector ret_types;
  // arg_nodes[i] contains the i'th function input. In other words,
  // GetNodeAttr(arg_nodes[i]->attrs(), "index") == i.
  absl::InlinedVector<Node*, 4UL> arg_nodes;
  // ret_nodes[i] contains the i'th function output. In other words,
  // GetNodeAttr(ret_nodes[i]->attrs(), "index") == i.
  absl::InlinedVector<Node*, 4UL> ret_nodes;
  absl::InlinedVector<Node*, 4UL> control_ret_nodes;

  // Allocator attributes arg/ret nodes of the function body.
  absl::InlinedVector<AllocatorAttributes, 4UL> args_alloc_attrs;
  absl::InlinedVector<AllocatorAttributes, 4UL> rets_alloc_attrs;

  FunctionBody() {}
  FunctionBody(core::RefCountPtr<FunctionRecord>&& record,
               DataTypeSlice arg_types, DataTypeSlice ret_types, Graph* g);
  ~FunctionBody();

  // Finalizes the function body by unreferencing the function record,
  // destructing the graph it own, and resetting the node pointers. It populates
  // the alloc attrs for the function body, so that
  // FunctionLibraryRuntime::RunRemote can use it to allocate tensors.
  //
  // Returns an error if the allocator attributes cannot be populated.
  absl::Status Finalize();
};

}  // end namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_FUNCTION_BODY_H_
