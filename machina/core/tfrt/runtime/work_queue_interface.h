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
#ifndef MACHINA_CORE_TFRT_RUNTIME_WORK_QUEUE_INTERFACE_H_
#define MACHINA_CORE_TFRT_RUNTIME_WORK_QUEUE_INTERFACE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "machina/core/platform/context.h"
#include "machina/core/platform/statusor.h"
#include "machina/core/platform/threadpool_interface.h"
#include "machina/core/profiler/lib/connected_traceme.h"
#include "machina/core/profiler/lib/traceme_encode.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime

namespace machina {
namespace tfrt_stub {

// This is an intermediate interface in machina for injecting thread pool
// implementation into TFRT. We can add savedmodel/machina specific
// methods (eg. create an intra op thread pool) without changing TFRT core.
class WorkQueueInterface : public tfrt::ConcurrentWorkQueue {
 public:
  WorkQueueInterface() = default;
  explicit WorkQueueInterface(int64_t id) : id_(id) {}
  explicit WorkQueueInterface(int64_t id,
                              thread::ThreadPoolInterface* intra_op_threadpool)
      : id_(id), intra_op_threadpool_(intra_op_threadpool) {}
  ~WorkQueueInterface() override = 0;

  int64_t id() const { return id_; }

  thread::ThreadPoolInterface* GetIntraOpThreadPool() const {
    return intra_op_threadpool_;
  }

  // Returns per-request work queue if possible. A nullptr should be returned if
  // the implementation does not implement the per-request work queue.
  //
  // TODO(b/198671794): Remove per-request concepts from the work queue
  // interface so that the interface is more composable. Per-request logic
  // should be handled separately.
  ABSL_DEPRECATED("Create the instance directly instead.")
  virtual absl::StatusOr<std::unique_ptr<WorkQueueInterface>> InitializeRequest(
      int64_t request_id) const {
    return {nullptr};
  }

 private:
  int64_t id_ = 0;
  thread::ThreadPoolInterface* intra_op_threadpool_ = nullptr;
};

inline WorkQueueInterface::~WorkQueueInterface() = default;

// Creates a WorkQueueInterface from a ConcurrentWorkQueue. The returned
// WorkQueueInterface simply delegates all its public methods to the specified
// ConcurrentWorkQueue.
std::unique_ptr<WorkQueueInterface> WrapDefaultWorkQueue(
    std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue);

// Creates a WorkQueueInterface from a ConcurrentWorkQueue. The returned
// WorkQueueInterface simply delegates all its public methods to the specified
// ConcurrentWorkQueue. The `intra_thread_pool` is stored and will be passed out
// when `InitializeRequest()` is called.
std::unique_ptr<WorkQueueInterface> WrapDefaultWorkQueue(
    std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue,
    thread::ThreadPoolInterface* intra_thread_pool);

// A helper function that wraps tasks with traceme events.
template <typename Callable>
tfrt::TaskFunction WrapWork(int64_t id, absl::string_view name,
                            Callable&& work) {
  machina::Context context(machina::ContextKind::kThread);
  tsl::profiler::TraceMeProducer producer(
      [&]() { return absl::StrCat("producer_", name); },
      tsl::profiler::ContextType::kTfrtExecutor);
  return tfrt::TaskFunction([traceme_id = producer.GetContextId(),
                             name = std::string(name),
                             context = std::move(context),
                             work = std::forward<Callable>(work)]() mutable {
    tsl::profiler::TraceMeConsumer consumer(
        [&]() { return absl::StrCat("consumer_", name); },
        tsl::profiler::ContextType::kTfrtExecutor, traceme_id,
        tsl::profiler::TraceMeLevel::kInfo);
    machina::WithContext wc(context);
    std::forward<Callable>(work)();
  });
}

}  // namespace tfrt_stub
}  // namespace machina

#endif  // MACHINA_CORE_TFRT_RUNTIME_WORK_QUEUE_INTERFACE_H_
