/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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
#ifndef MACHINA_CORE_TFRT_IFRT_IFRT_RESTORE_TENSOR_REGISTRY_H_
#define MACHINA_CORE_TFRT_IFRT_IFRT_RESTORE_TENSOR_REGISTRY_H_

#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "machina/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "machina/xla/python/ifrt/future.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.pb.h"

namespace machina {
namespace ifrt_serving {

// This class is thread safe.
class IfrtRestoreTensorRegistry {
 public:
  struct RestoredTensorInfo {
    bool used_by_host = false;
    DtypeAndShape dtype_and_shape;
    xla::ifrt::Future<machina::Tensor> tensor_future;
  };
  // Tries to register a loaded variable with the given name.
  // Returns an error if the named tensor already exists.
  absl::Status TryRegister(absl::string_view name,
                           RestoredTensorInfo restored_tensor_info)
      ABSL_LOCKS_EXCLUDED(mutex_);

  xla::ifrt::Future<machina::Tensor> GetRestoredTensor(
      absl::string_view name) const ABSL_LOCKS_EXCLUDED(mutex_);

  // Sets the tensor as used by the host. To ensure a tensor's host memory
  // is released, this function must be called at least once before the Freeze.
  absl::Status SetUsedByHost(absl::string_view name)
      ABSL_LOCKS_EXCLUDED(mutex_);

  absl::StatusOr<DtypeAndShape> GetDtypeAndShape(absl::string_view name) const
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Part of freezing the model is to release the host tensors that are used by
  // the device only. The caller guarantees those tensors are already loaded to
  // the device.
  void Freeze() ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<std::string, RestoredTensorInfo> restored_tensors_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace ifrt_serving
}  // namespace machina

#endif  // MACHINA_CORE_TFRT_IFRT_IFRT_RESTORE_TENSOR_REGISTRY_H_
