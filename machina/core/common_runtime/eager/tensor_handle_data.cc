/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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
#include "machina/core/common_runtime/eager/tensor_handle_data.h"

#include <utility>
#include <variant>

#include "machina/core/common_runtime/eager/eager_executor.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/lib/strings/strcat.h"
#include "machina/core/profiler/lib/traceme.h"

namespace machina {

absl::Status LocalTensorHandleData::Tensor(const machina::Tensor** t) const {
  TF_RETURN_IF_ERROR(WaitReady("Tensor"));

  *t = &tensor_;

  return absl::OkStatus();
}

absl::Status LocalTensorHandleData::TensorValue(machina::TensorValue* t) {
  TF_RETURN_IF_ERROR(WaitReady("TensorValue"));

  machina::Tensor& tensor = tensor_;
  *t = machina::TensorValue(&tensor);

  return absl::OkStatus();
}

absl::Status LocalTensorHandleData::Shape(TensorShape* shape) const {
  TF_RETURN_IF_ERROR(WaitReady("Shape"));

  *shape = tensor_.shape();

  return absl::OkStatus();
}

absl::Status LocalTensorHandleData::NumDims(int* num_dims) const {
  TF_RETURN_IF_ERROR(WaitReady("NumDims"));

  *num_dims = tensor_.dims();

  return absl::OkStatus();
}

absl::Status LocalTensorHandleData::Dim(int dim_index, int64_t* dim) const {
  TF_RETURN_IF_ERROR(WaitReady("Dim"));

  *dim = tensor_.dim_size(dim_index);

  return absl::OkStatus();
}

absl::Status LocalTensorHandleData::NumElements(int64_t* num_elements) const {
  TF_RETURN_IF_ERROR(WaitReady("NumElements"));

  *num_elements = tensor_.NumElements();

  return absl::OkStatus();
}

absl::Status LocalTensorHandleData::Unprotect() {
  if (!IsReady()) {
    return errors::Internal("Cannot unprotect a non-ready tensor");
  }

  forwarding_protection_tensor_ = machina::Tensor();

  return absl::OkStatus();
}

absl::Status LocalTensorHandleData::SetTensor(machina::Tensor&& t) {
  DCHECK(!IsReady()) << "SetTensor is only called on non-ready handles.";

  tensor_ = std::move(t);
  // Create copy of original tensor to avoid forwarding
  forwarding_protection_tensor_ = tensor_;

  auto& state = std::get<BlockingControl>(ctrl_);
  state.SetReady();

  return absl::OkStatus();
}

string LocalTensorHandleData::DebugString() const {
  if (IsReady()) {
    return tensor_.DeviceSafeDebugString();
  } else {
    return "LocalTensorHandleData";
  }
}

void LocalTensorHandleData::BlockingControl::SetReady() {
  mutex_lock l(mu_);
  is_ready_ = true;
}

absl::Status LocalTensorHandleData::BlockingControl::WaitReady(
    const char* caller) const {
  tf_shared_lock l(mu_);
  if (!is_ready_) {
    tsl::profiler::TraceMe activity(
        [caller] { return absl::StrCat(caller, " WaitReady"); },

        tsl::profiler::TraceMeLevel::kInfo);
    DVLOG(3) << "WaitReady: " << caller << " " << this;
    mu_.Await(Condition(&is_ready_));
  }

  return is_poisoned_;
}

void LocalTensorHandleData::BlockingControl::Poison(absl::Status status) {
  mutex_lock l(mu_);
  if (is_ready_) {
    LOG(ERROR) << "Poison can only be called on non-ready handle: " << this;
    return;
  }
  is_poisoned_ = status;
  is_ready_ = true;
}

}  // namespace machina
