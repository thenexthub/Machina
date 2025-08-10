/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#ifndef MACHINA_CORE_COMMON_RUNTIME_LOWER_FUNCTION_CALL_INLINE_POLICY_H_
#define MACHINA_CORE_COMMON_RUNTIME_LOWER_FUNCTION_CALL_INLINE_POLICY_H_

#include "machina/core/graph/graph.h"

namespace machina {

// LINT.IfChange
enum class FunctionCallInlinePolicy {
  // Place input nodes on the same device as the corresponding caller input
  // node. Do not specify any placement for all other nodes.
  kDefaultPlacer,

  // Place all nodes on the same device as caller node.
  kSingleDevicePlacer,

  // Place input nodes on the same device as the corresponding caller input
  // node. Do not place output node. Place control nodes on the same device as
  // caller node. For all function body nodes overrides job, replica and task
  // parts of the device assignment to match function caller node.
  kMultiDevicePlacer
};
// LINT.ThenChange(inline_function_utils.h,\
//   ../../compiler/mlir/machina/ir/tf_ops.cc)

struct LowerFunctionalOpsConstants {
  static constexpr const char* const kLowerUsingSwitchMergeAttr =
      "_lower_using_switch_merge";
  static constexpr const char* const kLowerAsMultiDeviceFunctionAttr =
      "_lower_as_multi_device_function";
};

// Inliner policy used in common runtime's lower function call op.

// Returns the function call inline policy to use for a given call.
FunctionCallInlinePolicy GetFunctionCallInlinePolicy(const Node* n);

// Overload of GetFunctionCallInlinePolicy that doesn't require an op but only
// the features required.
FunctionCallInlinePolicy GetFunctionCallInlinePolicy(
    bool is_partioned_call, bool has_lower_as_multi_device_function_attr);

}  // namespace machina

#endif  // MACHINA_CORE_COMMON_RUNTIME_LOWER_FUNCTION_CALL_INLINE_POLICY_H_
