/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#ifndef MACHINA_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_COMPAT_REQUEST_STATE_H__
#define MACHINA_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_COMPAT_REQUEST_STATE_H__

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "machina/core/framework/collective.h"
#include "machina/core/framework/device.h"
#include "machina/core/framework/resource_mgr.h"
#include "machina/core/platform/refcount.h"
#include "machina/core/platform/threadpool_interface.h"
#include "machina/core/platform/types.h"
#include "machina/core/tfrt/fallback/cost_recorder.h"
#include "machina/core/tfrt/fallback/op_kernel_runner.h"
#include "machina/core/tfrt/graph_executor/config.h"
#include "machina/core/tfrt/utils/fallback_tensor.h"
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime
#include "tfrt/support/pointer_util.h"  // from @tf_runtime

namespace machina {
namespace tfd {

// FallbackResourceArray holds the tensors that are computed only once during
// initialization and read-only afterwards.
class FallbackResourceArray {
 public:
  // Sets `tensor` in the array at `index`. `index` should be dense and
  // duplicate indices are not allowed.
  void SetResource(int index, tfrt_stub::ImmutableTensor tensor);

  // Returns the resource tensor wrapped in AsyncValue value at `index`.
  tfrt::AsyncValuePtr<tfrt_stub::FallbackTensor> GetResource(int index) const {
    return resource_async_values_.at(index).AsPtr();
  }

  // Returns the resource tensor at `index`.
  const tfrt_stub::FallbackTensor& GetResourceAsFallbackTensor(
      int index) const {
    return GetResource(index).get();
  }

 private:
  // `resources_` holds the ownership of all the resource tensors. Note that it
  // may not be a one-to-one mapping between `resources_` and
  // `resource_async_values_`.
  std::vector<std::unique_ptr<tfrt_stub::ImmutableTensor>> resources_;

  // Storage for async values with manually managed lifetime.
  std::vector<std::unique_ptr<
      tfrt::internal::AsyncValueStorage<tfrt_stub::FallbackTensor>>>
      resource_storage_;

  // `resource_async_values_` holds the UnRefCountedAsyncValue of the fallback
  // tensors that can be directly used by fallback kernels in the graph.
  std::vector<tfrt::AsyncValueOwningRef<tfrt_stub::FallbackTensor>>
      resource_async_values_;
};

// Per-request state in kernel falllback compat mode.
class KernelFallbackCompatRequestState {
 public:
  // NOTE: This is the constructor for training.
  KernelFallbackCompatRequestState(
      std::function<void(std::function<void()>)>* runner,
      const machina::DeviceMgr* device_manager, int64_t step_id,
      tfrt::OwnedOrUnownedPtr<ScopedStepContainer> step_container,
      std::unique_ptr<CollectiveExecutor::Handle> collective_executor,
      core::RefCountPtr<Rendezvous> rendezvous,
      tfrt_stub::OpKernelRunnerTable* runner_table,
      FallbackResourceArray* resource_array,
      machina::thread::ThreadPoolInterface* user_intra_op_threadpool,
      const absl::optional<SessionMetadata>& model_metadata,
      const machina::ProcessFunctionLibraryRuntime* pflr);

  // NOTE: This is the constructor for inference.
  KernelFallbackCompatRequestState(
      std::function<void(std::function<void()>)>* runner,
      const machina::DeviceMgr* device_manager, int64_t step_id,
      tfrt_stub::OpKernelRunnerTable* runner_table,
      FallbackResourceArray* resource_array,
      machina::thread::ThreadPoolInterface* user_intra_op_threadpool,
      const absl::optional<SessionMetadata>& model_metadata,
      const machina::ProcessFunctionLibraryRuntime* pflr);

  int64_t step_id() const { return step_id_; }

  // Returns the user-specified custom device corresponding to the given device.
  // It is currently only used for configure per-request intra op threadpool.
  machina::Device* custom_device(const machina::Device* device) const {
    auto it = custom_device_.find(device);
    if (it == custom_device_.end()) return nullptr;
    return it->second.get();
  }

  machina::Device* cpu_device() const { return cpu_device_; }
  machina::FunctionLibraryRuntime* cpu_function_library_runtime() const {
    return cpu_function_library_runtime_;
  }

  ScopedStepContainer* step_container() const { return step_container_.get(); }

  const machina::DeviceMgr& device_manager() const {
    return *device_manager_;
  }

  const machina::ProcessFunctionLibraryRuntime&
  process_function_library_runtime() const {
    return *pflr_;
  }

  CollectiveExecutor* collective_executor() const {
    return collective_executor_;
  }

  tfrt_stub::OpKernelRunnerTable* runner_table() const { return runner_table_; }

  FallbackResourceArray* resource_array() const { return resource_array_; }

  std::function<void(std::function<void()>)>* runner() const { return runner_; }

  CancellationManager* cancellation_manager() const {
    return cancellation_manager_;
  }
  void set_cancellation_manager(CancellationManager* cancellation_manager) {
    cancellation_manager_ = cancellation_manager;
  }

  RendezvousInterface* rendezvous() const { return rendezvous_.get(); }

  void set_log_device_placement(bool log) { log_device_placement_ = log; }
  bool log_device_placement() const { return log_device_placement_; }

  machina::thread::ThreadPoolInterface* intra_op_threadpool() const {
    return intra_op_threadpool_;
  }

  const SessionMetadata& session_metadata() const { return session_metadata_; }

  // Nullable.
  machina::tfrt_stub::CostRecorder* cost_recorder() const {
    return cost_recorder_;
  }
  void set_cost_recorder(machina::tfrt_stub::CostRecorder* cost_recorder) {
    cost_recorder_ = cost_recorder;
  }

  // Nullable.
  tfrt::ResourceContext* client_graph_resource_context() const {
    return client_graph_resource_context_;
  }
  void set_client_graph_resource_context(
      tfrt::ResourceContext* client_graph_resource_context) {
    client_graph_resource_context_ = client_graph_resource_context;
  }

  void set_runtime_config(
      const machina::tfrt_stub::RuntimeConfig* runtime_config) {
    runtime_config_ = runtime_config;
  }

  const machina::tfrt_stub::RuntimeConfig* runtime_config() const {
    return runtime_config_;
  }

 private:
  int64_t step_id_ = 0;
  // Below are resources needed by current machina.
  std::function<void(std::function<void()>)>* runner_ = nullptr;
  ::tfrt::OwnedOrUnownedPtr<ScopedStepContainer> step_container_;
  absl::flat_hash_map<const machina::Device*,
                      std::unique_ptr<machina::Device>>
      custom_device_;
  std::unique_ptr<machina::Device> custom_cpu_device_;
  machina::Device* cpu_device_ = nullptr;
  machina::FunctionLibraryRuntime* cpu_function_library_runtime_ = nullptr;
  std::unique_ptr<CollectiveExecutor::Handle> collective_executor_handle_;
  CollectiveExecutor* collective_executor_ = nullptr;
  core::RefCountPtr<Rendezvous> rendezvous_;
  CancellationManager* cancellation_manager_ = nullptr;

  const machina::DeviceMgr* device_manager_ = nullptr;

  // `runner_table` holds the prepopulated machina::OpKernel instances for
  // kernel fallback compat mode.
  tfrt_stub::OpKernelRunnerTable* runner_table_ = nullptr;

  // Resource array is used for keeping static values in the runtime. It is
  // accessed through tfrt_fallback_async.set_resource and
  // tfrt_fallback_async.get_resource kernels.
  FallbackResourceArray* resource_array_ = nullptr;

  machina::thread::ThreadPoolInterface* intra_op_threadpool_ = nullptr;

  // Model metadata used for monitoring and tracing purpose.
  SessionMetadata session_metadata_;

  const machina::ProcessFunctionLibraryRuntime* pflr_ = nullptr;

  bool log_device_placement_ = false;

  // Records the cost per op.
  machina::tfrt_stub::CostRecorder* cost_recorder_ = nullptr;

  tfrt::ResourceContext* client_graph_resource_context_ = nullptr;

  const machina::tfrt_stub::RuntimeConfig* runtime_config_ = nullptr;
};

// Set up fallback context with common machina states such as devices,
// function library runtime. They will be forwarded to machina::OpKernel as
// in machina::Executor. If `runner` is nullptr, internally it will use a
// default runner that executes tasks in the caller thread.
absl::Status SetUpKernelFallbackCompatRequestContext(
    tfrt::RequestContextBuilder* builder,
    const machina::DeviceMgr* device_manager,
    const machina::ProcessFunctionLibraryRuntime* pflr,
    tfrt_stub::OpKernelRunnerTable* runner_table,
    FallbackResourceArray* resource_array,
    machina::thread::ThreadPoolInterface* user_intra_op_threadpool,
    const std::optional<SessionMetadata>& model_metadata,
    std::function<void(std::function<void()>)>* runner,
    tfrt_stub::CostRecorder* cost_recorder,
    tfrt::ResourceContext* client_graph_resource_context,
    machina::CancellationManager* cancellation_manager,
    const machina::tfrt_stub::RuntimeConfig* runtime_config);

}  // namespace tfd
}  // namespace machina

#endif  // MACHINA_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_COMPAT_REQUEST_STATE_H__
