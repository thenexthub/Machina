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

#include "machina/core/tfrt/ifrt/ifrt_loaded_variable_utils.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "machina/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "machina/xla/python/ifrt/array.h"
#include "machina/xla/python/ifrt/client.h"
#include "machina/xla/python/ifrt/future.h"
#include "machina/xla/tsl/concurrency/ref_count.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/xla/tsl/platform/threadpool.h"
#include "machina/core/framework/resource_handle.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/tfrt/ifrt/ifrt_loaded_variable_registry.h"
#include "machina/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include "machina/core/tfrt/ifrt/sharding_utils.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

namespace machina {
namespace ifrt_serving {

namespace {

absl::StatusOr<xla::ifrt::ArrayRef> LoadIfrtVariable(
    std::shared_ptr<xla::ifrt::Client> ifrt_client,
    const tsl::thread::ThreadPool& thread_pool,
    const machina::Tensor& variable,
    const VariableDeviceShardingConfig& sharding_config) {
  return machina::ifrt_serving::MakeArrayFromTensor(
      *ifrt_client, variable, sharding_config.device_ids,
      sharding_config.hlo_sharding, thread_pool);
}

}  // namespace

absl::StatusOr<ifrt_serving::DtypeAndShape> GetDtypeAndShape(
    const ResourceHandle& resource_handle) {
  const std::vector<DtypeAndPartialTensorShape>& dtype_and_partial_shapes =
      resource_handle.dtypes_and_shapes();

  if (dtype_and_partial_shapes.size() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expected 1 dtype and shape, got ", dtype_and_partial_shapes.size()));
  }
  ifrt_serving::DtypeAndShape dtype_and_shape;
  if (!dtype_and_partial_shapes.front().shape.AsTensorShape(
          &dtype_and_shape.shape)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to convert partial shape to full tensor shape: ",
                     dtype_and_partial_shapes.front().shape.DebugString()));
  }

  dtype_and_shape.dtype = dtype_and_partial_shapes.front().dtype;
  return dtype_and_shape;
}

std::string GetRuntimeNameFromVarHandle(const ResourceHandle& handle) {
  return absl::StrCat(handle.container(), "__", handle.name());
}

absl::Status AsyncLoadRestoredTensorAsIfrtLoadedVariable(
    absl::string_view tensor_name,
    std::shared_ptr<xla::ifrt::Client> ifrt_client,
    const tsl::thread::ThreadPool& thread_pool,
    const ifrt_serving::IfrtRestoreTensorRegistry& restore_tensor_registry,
    ifrt_serving::IfrtLoadedVariableRegistry& loaded_variable_registry,
    tfrt::ConcurrentWorkQueue* checkpoint_loader_queue,
    const VariableDeviceShardingConfig& sharding_config) {
  IfrtLoadedVariableRegistry::Key key{
      .device_ids = sharding_config.device_ids,
      .input_name = std::string(tensor_name),
      .hlo_sharding = sharding_config.hlo_sharding,
  };

  if (loaded_variable_registry.GetLoadedVariable(key).ok()) {
    VLOG(1) << "Found alread registered variable for " << tensor_name;
    return absl::OkStatus();
  }

  xla::ifrt::Future<machina::Tensor> restored_tensor_future =
      restore_tensor_registry.GetRestoredTensor(tensor_name);
  if (!restored_tensor_future.IsValid()) {
    return absl::InternalError(absl::StrCat(
        "LoadVariableOp: failed to fetch variable tensor: ", tensor_name));
  }
  auto loaded_variable_promise =
      xla::ifrt::Future<xla::ifrt::ArrayRef>::CreatePromise();
  auto loaded_variable_future =
      xla::ifrt::Future<xla::ifrt::ArrayRef>(loaded_variable_promise);
  TF_ASSIGN_OR_RETURN(
      absl::StatusOr<ifrt_serving::DtypeAndShape> dtype_and_shape,
      restore_tensor_registry.GetDtypeAndShape(tensor_name));
  TF_RETURN_IF_ERROR(loaded_variable_registry.TryRegisterLoadedVariable(
      key,
      [&]() -> absl::StatusOr<
                ifrt_serving::IfrtLoadedVariableRegistry::LoadedVariable> {
        return ifrt_serving::IfrtLoadedVariableRegistry::LoadedVariable(
            {.array = loaded_variable_future});
      }));

  restored_tensor_future.OnReady(
      [ifrt_client = std::move(ifrt_client), &thread_pool = thread_pool,
       checkpoint_loader_queue = checkpoint_loader_queue,
       sharding_config = sharding_config,
       loaded_variable_promise = std::move(loaded_variable_promise)](
          absl::StatusOr<machina::Tensor> restored_tensor) mutable {
        if (!restored_tensor.ok()) {
          loaded_variable_promise.Set(restored_tensor.status());
          return;
        }

        // Transfer tensor to array in a separate thread.
        checkpoint_loader_queue->AddTask(
            [ifrt_client = ifrt_client, &thread_pool = thread_pool,
             sharding_config = std::move(sharding_config),
             restored_tensor = std::move(*restored_tensor),
             loaded_variable_promise =
                 std::move(loaded_variable_promise)]() mutable {
              absl::StatusOr<xla::ifrt::ArrayRef> variable_array =
                  LoadIfrtVariable(ifrt_client, thread_pool, restored_tensor,
                                   sharding_config);
              loaded_variable_promise.Set(std::move(variable_array));
            });
      });
  return absl::OkStatus();
}

}  // namespace ifrt_serving
}  // namespace machina
