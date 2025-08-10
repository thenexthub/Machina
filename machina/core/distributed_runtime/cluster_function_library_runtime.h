/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#ifndef MACHINA_CORE_DISTRIBUTED_RUNTIME_CLUSTER_FUNCTION_LIBRARY_RUNTIME_H_
#define MACHINA_CORE_DISTRIBUTED_RUNTIME_CLUSTER_FUNCTION_LIBRARY_RUNTIME_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/types/optional.h"
#include "machina/core/distributed_runtime/worker_cache.h"
#include "machina/core/distributed_runtime/worker_interface.h"
#include "machina/core/framework/function.h"

namespace machina {

class WorkerSession;

// ClusterFunctionLibraryRuntime contains methods to Instantiate and Run
// functions across processes by making RPCs through worker service.
class ClusterFunctionLibraryRuntime : public DistributedFunctionLibraryRuntime {
 public:
  ClusterFunctionLibraryRuntime(WorkerSession* worker_session,
                                bool create_worker_session_called,
                                DeviceMgr* remote_device_mgr)
      : worker_session_(worker_session),
        create_worker_session_called_(create_worker_session_called),
        remote_device_mgr_(remote_device_mgr) {}

  ~ClusterFunctionLibraryRuntime() override;

  void Instantiate(const string& function_name,
                   const FunctionLibraryDefinition& lib_def, AttrSlice attrs,
                   const FunctionLibraryRuntime::InstantiateOptions& options,
                   FunctionLibraryRuntime::LocalHandle* handle,
                   FunctionLibraryRuntime::DoneCallback done) override;

  void Run(const FunctionLibraryRuntime::Options& opts,
           FunctionLibraryRuntime::LocalHandle handle,
           absl::Span<const Tensor> args, std::vector<Tensor>* rets,
           FunctionLibraryRuntime::DoneCallback done) override;

  void Run(const FunctionLibraryRuntime::Options& opts,
           FunctionLibraryRuntime::LocalHandle handle,
           absl::Span<const FunctionArg> args, std::vector<FunctionRet>* rets,
           FunctionLibraryRuntime::DoneCallback done) override;

  void CleanUp(uint64 step_id, FunctionLibraryRuntime::LocalHandle handle,
               FunctionLibraryRuntime::DoneCallback done) override;

  DeviceMgr* remote_device_mgr() const override { return remote_device_mgr_; }

 private:
  static absl::Status ConstructFunctionGraph(
      const OpDef& sig, AttrSlice attrs,
      const FunctionLibraryRuntime::InstantiateOptions& options,
      const FunctionLibraryDefinition& flib_def, GraphDef* g,
      std::vector<string>* send_keys, std::vector<string>* recv_keys);
  friend class ClusterFunctionLibraryRuntimeTest;

  mutable mutex mu_;
  WorkerSession* const worker_session_ = nullptr;  // not owned.
  const bool create_worker_session_called_;

  DeviceMgr* remote_device_mgr_;  // not owned.

  struct FunctionData {
    const string graph_handle;
    const string target;
    // Hold a shared pointer to the underlying worker cache to avoid it being
    // deleted in potential cluster update.
    const std::shared_ptr<WorkerCacheInterface> worker_cache;
    WorkerInterface* wi = nullptr;
    const std::vector<string> send_keys;
    const std::vector<string> recv_keys;

    FunctionData(const string& graph_handle, const string& target,
                 std::shared_ptr<WorkerCacheInterface> worker_cache,
                 WorkerInterface* wi, const std::vector<string>& send_keys,
                 const std::vector<string>& recv_keys)
        : graph_handle(graph_handle),
          target(target),
          worker_cache(std::move(worker_cache)),
          wi(wi),
          send_keys(send_keys),
          recv_keys(recv_keys) {}
  };

  std::vector<FunctionData> function_data_ TF_GUARDED_BY(mu_);
};

}  // namespace machina

#endif  // MACHINA_CORE_DISTRIBUTED_RUNTIME_CLUSTER_FUNCTION_LIBRARY_RUNTIME_H_
