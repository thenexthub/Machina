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

#include "machina/c/experimental/saved_model/core/ops/variable_ops.h"

#include <cstdint>
#include <cstring>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "machina/c/eager/abstract_tensor_handle.h"
#include "machina/c/eager/immediate_execution_context.h"
#include "machina/c/eager/immediate_execution_operation.h"
#include "machina/c/eager/immediate_execution_tensor_handle.h"
#include "machina/core/framework/resource_handle.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/lib/gtl/inlined_vector.h"
#include "machina/core/lib/llvm_rtti/llvm_rtti.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/status.h"
#include "tsl/platform/errors.h"

namespace machina {
namespace internal {

absl::Status CreateUninitializedResourceVariable(
    ImmediateExecutionContext* ctx, DataType dtype, TensorShape shape,
    const char* raw_device_name, ImmediateTensorHandlePtr* handle) {
  ImmediateOpPtr varhandle_op(ctx->CreateOperation());

  TF_RETURN_IF_ERROR(varhandle_op->Reset("VarHandleOp", raw_device_name));
  TF_RETURN_IF_ERROR(varhandle_op->SetAttrType("dtype", dtype));

  // Note that if shape is unknown rank, shape.dim_sizes() will be empty, and
  // shape.dims() will be -1.
  absl::InlinedVector<int64_t, 4UL> dim_sizes = shape.dim_sizes();
  TF_RETURN_IF_ERROR(varhandle_op->SetAttrShape(
      "shape", reinterpret_cast<const int64_t*>(dim_sizes.data()),
      shape.dims()));
  TF_RETURN_IF_ERROR(varhandle_op->SetAttrString("container", "", 0));
  TF_RETURN_IF_ERROR(
      varhandle_op->SetAttrString("shared_name", ResourceHandle::ANONYMOUS_NAME,
                                  strlen(ResourceHandle::ANONYMOUS_NAME)));

  AbstractTensorHandle* var_handle = nullptr;
  int num_retvals = 1;
  TF_RETURN_IF_ERROR(varhandle_op->Execute(
      absl::MakeSpan(&var_handle, num_retvals), &num_retvals));
  AbstractTensorHandlePtr owned_var_handle(var_handle);
  if (!machina::isa<ImmediateExecutionTensorHandle>(
          owned_var_handle.get())) {
    return errors::Internal("Unexpected tensor handle kind.");
  }
  handle->reset(reinterpret_cast<ImmediateExecutionTensorHandle*>(
      owned_var_handle.release()));
  return absl::Status();
}

absl::Status AssignVariable(ImmediateExecutionContext* ctx,
                            ImmediateExecutionTensorHandle* variable_handle,
                            DataType dtype,
                            ImmediateExecutionTensorHandle* value) {
  ImmediateOpPtr assign_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(assign_op->Reset("AssignVariableOp", nullptr));
  TF_RETURN_IF_ERROR(assign_op->SetAttrType("dtype", dtype));
  TF_RETURN_IF_ERROR(assign_op->AddInput(variable_handle));
  TF_RETURN_IF_ERROR(assign_op->AddInput(value));

  int num_retvals = 0;
  TF_RETURN_IF_ERROR(assign_op->Execute({}, &num_retvals));
  return absl::Status();
}

absl::Status ReadVariable(ImmediateExecutionContext* ctx,
                          ImmediateExecutionTensorHandle* variable_handle,
                          DataType dtype, ImmediateTensorHandlePtr* output) {
  ImmediateOpPtr read_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(read_op->Reset("ReadVariableOp", nullptr));
  TF_RETURN_IF_ERROR(read_op->SetAttrType("dtype", dtype));
  TF_RETURN_IF_ERROR(read_op->AddInput(variable_handle));

  AbstractTensorHandle* value = nullptr;
  int num_retvals = 1;
  TF_RETURN_IF_ERROR(
      read_op->Execute(absl::MakeSpan(&value, num_retvals), &num_retvals));
  AbstractTensorHandlePtr owned_value(value);
  if (!machina::isa<ImmediateExecutionTensorHandle>(owned_value.get())) {
    return errors::Internal("Unexpected tensor handle kind.");
  }
  output->reset(
      reinterpret_cast<ImmediateExecutionTensorHandle*>(owned_value.release()));
  return absl::Status();
}

absl::Status DestroyResource(ImmediateExecutionContext* ctx,
                             ImmediateExecutionTensorHandle* handle) {
  ImmediateOpPtr destroy_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(destroy_op->Reset("DestroyResourceOp", nullptr));
  TF_RETURN_IF_ERROR(destroy_op->SetAttrBool("ignore_lookup_error", true));
  TF_RETURN_IF_ERROR(destroy_op->AddInput(handle));

  int num_retvals = 0;
  TF_RETURN_IF_ERROR(destroy_op->Execute({}, &num_retvals));
  return absl::Status();
}

}  // namespace internal
}  // namespace machina
