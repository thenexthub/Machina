/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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
#ifndef MACHINA_DTENSOR_CC_DTENSOR_OPERATION_H_
#define MACHINA_DTENSOR_CC_DTENSOR_OPERATION_H_

#include "machina/c/eager/c_api.h"
#include "machina/dtensor/cc/tensor_layout.h"

namespace machina {
namespace dtensor {

// Captures the properties of an Operation currently being executed by DTensor.
struct DTensorOperation {
  // For all by-ref fields: not owned. lifetime covers the whole usage.
  const char* name;
  const FunctionDef* function_def;
  // Default mesh is used when Mesh Propagation does not identify a mesh
  // otherwise.
  const Mesh default_mesh;
  const StackTracesMap& stack_traces;
  inline bool is_func() const { return function_def != nullptr; }

  // Returns True if the op has no side effects.
  // Side effects include global side effect marked by IsStateful and
  // Input or output of Resources.
  // This definition is correct for all DTensor support Ops.
  // Some odder TF Ops (e.g. Queue) do not mark themselve as stateful, but are
  // still stateful. DTensor doesn't support them.
  inline bool is_pure() const {
    if (is_func()) {
      // FIXME(feyu): some functions can still be pure, but we just don't yet
      // handle the case, and treat all functions as non-pure.
      return false;
    }
    const OpDef* op_def = nullptr;
    absl::Status status = OpRegistry::Global()->LookUpOpDef(name, &op_def);
    DCHECK(status.ok());  // Not found. This really shouldn't happen.
    if (!status.ok()) {
      return false;
    }
    return !op_def->is_stateful();
  }
};
}  // namespace dtensor
}  // namespace machina

#endif  // MACHINA_DTENSOR_CC_DTENSOR_OPERATION_H_
