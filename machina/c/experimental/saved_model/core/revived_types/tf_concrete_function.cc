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

#include "machina/c/experimental/saved_model/core/revived_types/tf_concrete_function.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "machina/c/eager/abstract_tensor_handle.h"
#include "machina/c/eager/immediate_execution_operation.h"
#include "machina/c/eager/immediate_execution_tensor_handle.h"
#include "machina/c/experimental/saved_model/core/function_metadata.h"
#include "machina/c/experimental/saved_model/core/revived_types/flat_tensor_function.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/platform/status.h"
#include "machina/core/protobuf/saved_object_graph.pb.h"
#include "machina/core/protobuf/struct.pb.h"
#include "tsl/platform/errors.h"

namespace machina {

TFConcreteFunction::TFConcreteFunction(std::unique_ptr<FlatTensorFunction> func,
                                       FunctionMetadata metadata)
    : func_(std::move(func)), metadata_(std::move(metadata)) {}

absl::Status TFConcreteFunction::Create(
    const FunctionDef* function_def,
    std::vector<ImmediateExecutionTensorHandle*> captures,
    FunctionMetadata metadata, ImmediateExecutionContext* ctx,
    std::unique_ptr<TFConcreteFunction>* out) {
  std::unique_ptr<FlatTensorFunction> func;
  TF_RETURN_IF_ERROR(FlatTensorFunction::Create(
      function_def, std::move(captures), ctx, &func));

  out->reset(new TFConcreteFunction(std::move(func), std::move(metadata)));
  return absl::Status();
}

const FunctionMetadata& TFConcreteFunction::GetFunctionMetadata() const {
  return metadata_;
}

absl::Status TFConcreteFunction::MakeCallOp(
    absl::Span<AbstractTensorHandle* const> inputs, ImmediateOpPtr* out) const {
  return func_->MakeCallOp(inputs, out);
}

}  // namespace machina
