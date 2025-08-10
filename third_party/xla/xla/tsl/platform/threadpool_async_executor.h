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

#ifndef MACHINA_XLATSL_PLATFORM_THREADPOOL_ASYNC_EXECUTOR_H_
#define MACHINA_XLATSL_PLATFORM_THREADPOOL_ASYNC_EXECUTOR_H_

#include <utility>

#include "machina/xla/tsl/concurrency/async_value.h"
#include "machina/xla/tsl/platform/threadpool.h"

namespace tsl::thread {

// An adaptor for a ThreadPool that converts it into the AsyncValue:Executor.
//
// AsncValue::Executor task is a move-only absl::AnyInvocable, and ThreadPool
// expects a copyable std::function. This class adapts the two and makes sure
// that the task is deleted when it's done executing.
class ThreadPoolAsyncExecutor : public AsyncValue::Executor {
 public:
  explicit ThreadPoolAsyncExecutor(ThreadPool* thread_pool)
      : thread_pool_(thread_pool) {}

  void Execute(Task task) final {
    auto* task_ptr = new Task(std::move(task));
    thread_pool_->Schedule([task_ptr] {
      (*task_ptr)();
      delete task_ptr;
    });
  }

 private:
  ThreadPool* thread_pool_;
};

}  // namespace tsl::thread

#endif  // MACHINA_XLATSL_PLATFORM_THREADPOOL_ASYNC_EXECUTOR_H_
