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

#include "machina/c/experimental/saved_model/core/ops/restore_ops.h"

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "machina/c/eager/abstract_tensor_handle.h"
#include "machina/c/eager/immediate_execution_context.h"
#include "machina/c/eager/immediate_execution_operation.h"
#include "machina/c/eager/immediate_execution_tensor_handle.h"
#include "machina/c/tensor_interface.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/lib/llvm_rtti/llvm_rtti.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/tstring.h"
#include "tsl/platform/errors.h"

namespace machina {
namespace internal {

namespace {

// Creates a scalar string tensorhandle containing a single string `s`
absl::Status CreateStringScalarTensorHandle(ImmediateExecutionContext* ctx,
                                            const std::string& s,
                                            ImmediateTensorHandlePtr* out) {
  AbstractTensorPtr tensor(ctx->CreateStringScalar(s));
  if (tensor.get() == nullptr) {
    return errors::Internal(
        "Failed to create scalar string tensor for checkpoint restore");
  }

  out->reset(ctx->CreateLocalHandle(tensor.get()));
  return absl::Status();
}

// Creates a Rank 1 string tensorhandle containing a single string `s`
absl::Status CreateStringVectorTensorHandle(ImmediateExecutionContext* ctx,
                                            const std::string& s,
                                            ImmediateTensorHandlePtr* out) {
  int64_t flat_shape[] = {1};
  AbstractTensorPtr tensor(ctx->CreateTensor(DT_STRING, flat_shape));
  if (tensor.get() == nullptr) {
    return errors::Internal(
        "Failed to create vector string tensor for checkpoint restore");
  }
  // Use placement new to construct the string, since we don't have
  // access to Tensor::flat. This is conceptually equivalent to:
  // tensor.flat<tstring>()(0) = s
  new (tensor->Data()) tstring(s);

  out->reset(ctx->CreateLocalHandle(tensor.get()));
  return absl::Status();
}

}  // namespace

absl::Status SingleRestore(ImmediateExecutionContext* ctx,
                           const std::string& prefix,
                           const std::string& checkpoint_key, DataType dtype,
                           ImmediateTensorHandlePtr* out) {
  // Create the EagerOp
  ImmediateOpPtr restore_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(restore_op->Reset("RestoreV2", "/cpu:0"));
  TF_RETURN_IF_ERROR(restore_op->SetAttrTypeList("dtypes", &dtype, 1));

  ImmediateTensorHandlePtr prefix_handle;
  TF_RETURN_IF_ERROR(
      CreateStringScalarTensorHandle(ctx, prefix, &prefix_handle));

  ImmediateTensorHandlePtr names_handle;
  TF_RETURN_IF_ERROR(
      CreateStringVectorTensorHandle(ctx, checkpoint_key, &names_handle));

  // Note that empty string is the slice spec used for a non-partitioned
  // ResourceVariable:
  // https://github.com/machina/machina/blob/06ff30f7ea35098cb68a231a9eb7ff3ff4be4e1e/machina/python/training/saving/saveable_object_util.py#L194
  ImmediateTensorHandlePtr shapes_and_slices_handle;
  TF_RETURN_IF_ERROR(
      CreateStringVectorTensorHandle(ctx, "", &shapes_and_slices_handle));

  TF_RETURN_IF_ERROR(restore_op->AddInput(prefix_handle.get()));
  TF_RETURN_IF_ERROR(restore_op->AddInput(names_handle.get()));
  TF_RETURN_IF_ERROR(restore_op->AddInput(shapes_and_slices_handle.get()));

  AbstractTensorHandle* restored_handle = nullptr;
  int num_retvals = 1;
  TF_RETURN_IF_ERROR(restore_op->Execute(
      absl::MakeSpan(&restored_handle, num_retvals), &num_retvals));
  AbstractTensorHandlePtr owned_restored_handle(restored_handle);
  if (!machina::isa<ImmediateExecutionTensorHandle>(
          owned_restored_handle.get())) {
    return errors::Internal("Unexpected tensor handle kind.");
  }
  out->reset(reinterpret_cast<ImmediateExecutionTensorHandle*>(
      owned_restored_handle.release()));
  return absl::Status();
}

}  // namespace internal
}  // namespace machina
