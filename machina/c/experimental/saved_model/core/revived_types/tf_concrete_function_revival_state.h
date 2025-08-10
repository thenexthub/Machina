/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#ifndef MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_TF_CONCRETE_FUNCTION_REVIVAL_STATE_H_
#define MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_TF_CONCRETE_FUNCTION_REVIVAL_STATE_H_

#include <memory>
#include <vector>

#include "absl/types/optional.h"
#include "machina/c/eager/immediate_execution_context.h"
#include "machina/c/eager/immediate_execution_tensor_handle.h"
#include "machina/c/experimental/saved_model/core/revived_types/tf_concrete_function.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/protobuf/saved_object_graph.pb.h"

namespace machina {

// TFConcreteFunctionRevivalState wraps the state needed for building a
// TF_ConcreteFunction. This is mainly used in PartiallyRevivedObjects, which
// wraps partially constructed Function and Resource objects.
struct TFConcreteFunctionRevivalState {
  // Index of the node in the SavedObjectGraph it was loaded from.
  int node_id;

  // Pointer to the original functiondef. fdef_ is guaranteed to be
  // non-null.
  const FunctionDef* fdef;

  // TensorHandle captures for this funtion
  std::vector<ImmediateExecutionTensorHandle*> captures;

  // SavedConcreteFunction contains much of the metadata of the expected "types"
  // of the inputs and outputs of a function.
  // Note(bmzhao): saved_concrete_func_ is guaranteed to be non-null.
  const SavedConcreteFunction* saved_concrete_func;

  // This field is only present on TF2 ConcreteFunctions, and is useful for
  // determining the original argument *names* of the function, (since the
  // "canonicalized_input_signature" may append extra uniquifying integers).
  // However, SavedBareConcreteFunctions do not have a FunctionSpec.
  // Note(bmzhao): if function_spec_.has_value(), *function_spec_ is guaranteed
  // to be non-null.
  absl::optional<const FunctionSpec*> function_spec;
};

}  // namespace machina

#endif  // MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_TF_CONCRETE_FUNCTION_REVIVAL_STATE_H_
