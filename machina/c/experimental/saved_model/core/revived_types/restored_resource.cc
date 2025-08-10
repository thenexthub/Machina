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

#include "machina/c/experimental/saved_model/core/revived_types/restored_resource.h"

#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "machina/c/eager/abstract_tensor_handle.h"
#include "machina/c/eager/immediate_execution_operation.h"
#include "machina/c/eager/immediate_execution_tensor_handle.h"
#include "machina/c/experimental/saved_model/core/revived_types/tensorhandle_convertible.h"
#include "machina/c/experimental/saved_model/core/revived_types/tf_concrete_function.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/status.h"
#include "tsl/platform/errors.h"

namespace machina {

namespace {

absl::Status ExecuteNoArgDummyReturnFunction(TFConcreteFunction* func) {
  ImmediateOpPtr function_op;
  TF_RETURN_IF_ERROR(func->MakeCallOp({}, &function_op));

  AbstractTensorHandle* dummy_output = nullptr;
  int num_retvals = 1;
  TF_RETURN_IF_ERROR(function_op->Execute(
      absl::MakeSpan(&dummy_output, num_retvals), &num_retvals));
  AbstractTensorHandlePtr owned_dummy_output(dummy_output);
  return absl::Status();
}

}  // namespace

RestoredResource::RestoredResource(const std::string& device,
                                   TFConcreteFunction* create_resource,
                                   TFConcreteFunction* initialize,
                                   TFConcreteFunction* destroy_resource,
                                   ImmediateTensorHandlePtr resource_handle)
    : TensorHandleConvertible(std::move(resource_handle)),
      device_(device),
      create_resource_(create_resource),
      initialize_(initialize),
      destroy_resource_(destroy_resource) {}

absl::Status RestoredResource::Initialize() const {
  return ExecuteNoArgDummyReturnFunction(initialize_);
}

RestoredResource::~RestoredResource() {
  // Note(bmzhao): SavedModels saved before
  // https://github.com/machina/machina/commit/3c806101f57768e479f8646e7518bbdff1632ca3
  // did not have their destroy_resource function saved, meaning they will
  // leak resources.
  //
  // Check that handle is null before calling destroy_resource function in case
  // destructor is invoked unintentionally.
  if (destroy_resource_ != nullptr && handle() != nullptr) {
    absl::Status status = ExecuteNoArgDummyReturnFunction(destroy_resource_);
    if (!status.ok()) {
      LOG(WARNING)
          << "Failed executing destroy_resource function for RestoredResource: "
          << status.message();
    }
  }
}

}  // namespace machina
