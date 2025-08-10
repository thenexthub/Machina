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
#ifndef MACHINA_CORE_TFRT_RUNTIME_TF_THREADPOOL_CONCURRENT_WORK_QUEUE_H_
#define MACHINA_CORE_TFRT_RUNTIME_TF_THREADPOOL_CONCURRENT_WORK_QUEUE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/base/attributes.h"
#include "absl/status/statusor.h"
#include "machina/core/platform/cpu_info.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/threadpool_interface.h"
#include "machina/core/tfrt/runtime/work_queue_interface.h"
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/task_function.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace machina {
namespace tfrt_stub {

// This class defines a work queue based on the WorkQueueInterface that uses the
// Tensorflow threadpools to execute inter-op and intra-op closures.
class TfThreadPoolWorkQueue : public WorkQueueInterface {
 public:
  TfThreadPoolWorkQueue(
      machina::thread::ThreadPoolInterface* intra_op_threadpool,
      machina::thread::ThreadPoolInterface* inter_op_threadpool)
      : TfThreadPoolWorkQueue(/*id=*/0, intra_op_threadpool,
                              inter_op_threadpool) {}

  TfThreadPoolWorkQueue(
      int64_t id, machina::thread::ThreadPoolInterface* intra_op_threadpool,
      machina::thread::ThreadPoolInterface* inter_op_threadpool)
      : WorkQueueInterface(id, intra_op_threadpool),
        intra_op_threadpool_(intra_op_threadpool),
        inter_op_threadpool_(inter_op_threadpool) {}

  absl::StatusOr<std::unique_ptr<WorkQueueInterface>> InitializeRequest(
      int64_t request_id) const override;

  int GetParallelismLevel() const override {
    return inter_op_threadpool_->NumThreads();
  }
  std::string name() const override { return "TfThreadPoolWorkQueue"; }

  void AddTask(tfrt::TaskFunction work) override;

  std::optional<tfrt::TaskFunction> AddBlockingTask(
      tfrt::TaskFunction work, bool allow_queuing) override;

  ABSL_DEPRECATED("Use the destructor instead.")
  void Quiesce() override;

  void Await(
      tfrt::ArrayRef<::tfrt::RCReference<::tfrt::AsyncValue>> values) override;

  bool IsInWorkerThread() const override;

 private:
  machina::thread::ThreadPoolInterface* intra_op_threadpool_ = nullptr;
  machina::thread::ThreadPoolInterface* inter_op_threadpool_ = nullptr;
};

// Create a default TfThreadPoolWorkQueue that is implemented by
// machina::thread::ThreadPool. `num_inter_op_threads` and
// `num_intra_op_threads` must be larger than zero.
std::unique_ptr<TfThreadPoolWorkQueue> CreateDefaultTfThreadPoolWorkQueue(
    int num_inter_op_threads, int num_intra_op_threads);

}  // namespace tfrt_stub
}  // namespace machina

#endif  // MACHINA_CORE_TFRT_RUNTIME_TF_THREADPOOL_CONCURRENT_WORK_QUEUE_H_
