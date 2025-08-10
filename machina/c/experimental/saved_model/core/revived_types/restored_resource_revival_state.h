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

#ifndef MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_RESTORED_RESOURCE_REVIVAL_STATE_H_
#define MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_RESTORED_RESOURCE_REVIVAL_STATE_H_

#include <string>

#include "machina/c/eager/immediate_execution_tensor_handle.h"
#include "machina/c/experimental/saved_model/core/revived_types/tf_concrete_function_revival_state.h"

namespace machina {

// All "Resources" should have these 3 saved functions:
// https://github.com/machina/machina/blob/86dc281333d7d277ddc1882f2bca4b17e7ec40e5/machina/python/training/tracking/tracking.py#L277-L281
struct RestoredResourceRevivalState {
  std::string device;
  TFConcreteFunctionRevivalState* create_resource = nullptr;
  TFConcreteFunctionRevivalState* initialize = nullptr;
  TFConcreteFunctionRevivalState* destroy_resource = nullptr;
  ImmediateTensorHandlePtr resource_handle = nullptr;
};

}  // namespace machina

#endif  // MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_RESTORED_RESOURCE_REVIVAL_STATE_H_
