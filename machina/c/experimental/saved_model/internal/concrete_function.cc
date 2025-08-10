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

#include "machina/c/experimental/saved_model/public/concrete_function.h"

#include <cstddef>

#include "absl/types/span.h"
#include "machina/c/eager/abstract_tensor_handle.h"
#include "machina/c/eager/immediate_execution_operation.h"
#include "machina/c/eager/tfe_op_internal.h"
#include "machina/c/eager/tfe_tensorhandle_internal.h"
#include "machina/c/experimental/saved_model/core/concrete_function.h"
#include "machina/c/experimental/saved_model/core/function_metadata.h"
#include "machina/c/experimental/saved_model/internal/concrete_function_type.h"
#include "machina/c/experimental/saved_model/internal/function_metadata_type.h"
#include "machina/c/tf_status_internal.h"
#include "machina/core/platform/status.h"

extern "C" {

TF_FunctionMetadata* TF_ConcreteFunctionGetMetadata(TF_ConcreteFunction* func) {
  return machina::wrap(const_cast<machina::FunctionMetadata*>(
      &machina::unwrap(func)->GetFunctionMetadata()));
}

TFE_Op* TF_ConcreteFunctionMakeCallOp(TF_ConcreteFunction* func,
                                      TFE_TensorHandle** inputs, int num_inputs,
                                      TF_Status* status) {
  machina::ImmediateOpPtr call_op;
  absl::Span<machina::AbstractTensorHandle* const> input_span(
      reinterpret_cast<machina::AbstractTensorHandle**>(
          machina::unwrap(inputs)),
      static_cast<size_t>(num_inputs));
  status->status = machina::unwrap(func)->MakeCallOp(input_span, &call_op);
  if (!status->status.ok()) {
    return nullptr;
  }
  return machina::wrap(call_op.release());
}

}  // end extern "C"
