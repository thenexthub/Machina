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

#ifndef MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_TF_CONCRETE_FUNCTION_H_
#define MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_TF_CONCRETE_FUNCTION_H_

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "machina/c/eager/abstract_tensor_handle.h"
#include "machina/c/eager/immediate_execution_context.h"
#include "machina/c/eager/immediate_execution_operation.h"
#include "machina/c/eager/immediate_execution_tensor_handle.h"
#include "machina/c/experimental/saved_model/core/concrete_function.h"
#include "machina/c/experimental/saved_model/core/function_metadata.h"
#include "machina/c/experimental/saved_model/core/revived_types/flat_tensor_function.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/platform/status.h"
#include "machina/core/protobuf/saved_object_graph.pb.h"

namespace machina {

// TF Eager Runtime-based implementation of a "ConcreteFunction" loaded from a
// saved model.
class TFConcreteFunction : public ConcreteFunction {
 public:
  // Factory function for creating a TFConcreteFunction.
  //
  // Params:
  //  function_def - The function_def associated with the created
  //                 TFConcreteFunction. TFConcreteFunction will register this
  //                 function_def with `ctx` on creation, and de-register it on
  //                 destruction. function_def must be non-null, but
  //                 otherwise has no lifetime requirements.
  //  captures - The captured TensorHandles associated with this
  //             TFConcreteFunction.
  //  metadata - The FunctionMetadata associated with this TFConcreteFunction.
  //  ctx      - A handle to the Tensorflow runtime. This MUST be non-null and
  //             outlive TFConcreteFunction.
  //  out      - The output TFConcreteFunction.
  static absl::Status Create(
      const FunctionDef* function_def,
      std::vector<ImmediateExecutionTensorHandle*> captures,
      FunctionMetadata metadata, ImmediateExecutionContext* ctx,
      std::unique_ptr<TFConcreteFunction>* out);

  // This method returns the "Call" Op used to execute the function.
  absl::Status MakeCallOp(absl::Span<AbstractTensorHandle* const> inputs,
                          ImmediateOpPtr* out) const override;

  const FunctionMetadata& GetFunctionMetadata() const override;

  ~TFConcreteFunction() override = default;

 private:
  TFConcreteFunction(std::unique_ptr<FlatTensorFunction> func,
                     FunctionMetadata metadata);

  TFConcreteFunction(const TFConcreteFunction&) = delete;
  TFConcreteFunction& operator=(const TFConcreteFunction&) = delete;

  std::unique_ptr<FlatTensorFunction> func_;
  FunctionMetadata metadata_;
};

}  // namespace machina

#endif  // MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_TF_CONCRETE_FUNCTION_H_
