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

#ifndef MACHINA_CORE_KERNELS_BATCHING_UTIL_BOUNDED_EXECUTOR_H_
#define MACHINA_CORE_KERNELS_BATCHING_UTIL_BOUNDED_EXECUTOR_H_

#include <string>

#include "machina/core/platform/status.h"
#include "machina/core/platform/statusor.h"
#include "machina/core/platform/thread_annotations.h"
#include "machina/core/platform/threadpool.h"
#include "machina/core/platform/threadpool_interface.h"

namespace machina {
namespace serving {
// BoundedExecutor has a bounded number of threads and unlimited queue length,
// scheduled tasks are executed in a FIFO way.
class BoundedExecutor : public thread::ThreadPoolInterface {
 public:
  struct Options {
    Env* env = Env::Default();
    ThreadOptions thread_options;
    std::string thread_name;
    int num_threads = -1;
  };

  static absl::StatusOr<std::unique_ptr<BoundedExecutor>> Create(
      const Options& options);

  // Destructor. All threads will be joined.
  ~BoundedExecutor() override;

  // Enqueue a function to be executed.
  //
  // Callers are responsible to guarantee `func` is not nullptr.
  void Schedule(std::function<void()> func) override;

  // Returns the number of threads.
  int NumThreads() const override;

  int CurrentThreadId() const override;

 private:
  explicit BoundedExecutor(const Options& options);

  // Starts N workers (N == num_threads), polling tasks from `work_queue_`.
  void InitWorker();

  // A loop to fetch task from `work_queue_` and execute task.
  void Run();

  const Options& options_;

  mutex work_queue_mu_;
  std::deque<std::function<void()>> work_queue_ TF_GUARDED_BY(work_queue_mu_);
  condition_variable work_queue_cv_ TF_GUARDED_BY(work_queue_mu_);

  // A fixed number of threads.
  std::vector<std::unique_ptr<Thread>> threads_;
  BoundedExecutor(const BoundedExecutor&) = delete;
  void operator=(const BoundedExecutor&) = delete;
};

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_BATCHING_UTIL_BOUNDED_EXECUTOR_H_
