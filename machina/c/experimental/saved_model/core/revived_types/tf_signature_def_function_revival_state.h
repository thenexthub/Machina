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

#ifndef MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_TF_SIGNATURE_DEF_FUNCTION_REVIVAL_STATE_H_
#define MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_TF_SIGNATURE_DEF_FUNCTION_REVIVAL_STATE_H_

#include <memory>
#include <vector>

#include "absl/types/optional.h"
#include "machina/c/eager/immediate_execution_tensor_handle.h"
#include "machina/c/experimental/saved_model/core/revived_types/tf_signature_def_function.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/protobuf/saved_object_graph.pb.h"

namespace machina {

// FunctionBuilder wraps the state needed for building a SignatureDefFunction.
// This is mainly used in PartiallyRevivedObjects, which wraps partially
// constructed Function and Resource objects.
struct TFSignatureDefFunctionRevivalState {
  // Index of the node in the SavedObjectGraph it was loaded from.
  int node_id = 0;

  // Pointer to the original functiondef. fdef_ is guaranteed to be
  // non-null.
  const FunctionDef* fdef = nullptr;

  // SavedConcreteFunction contains much of the metadata of the expected "types"
  // of the inputs and outputs of a function.
  // Note(bmzhao): saved_concrete_func_ is guaranteed to be non-null.
  const SavedConcreteFunction* saved_concrete_func = nullptr;

  // The name of the SignatureDef key.
  std::string signature_key;

  // TensorHandle captures for this funtion
  std::vector<ImmediateExecutionTensorHandle*> captures;
};

}  // namespace machina

#endif  // MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_TF_SIGNATURE_DEF_FUNCTION_REVIVAL_STATE_H_
