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
#ifndef MACHINA_CORE_DATA_UNBOUNDED_THREAD_POOL_H_
#define MACHINA_CORE_DATA_UNBOUNDED_THREAD_POOL_H_

#include <deque>
#include <functional>
#include <memory>
#include <vector>

#include "machina/core/framework/thread_factory.h"
#include "machina/core/lib/core/notification.h"
#include "machina/core/lib/core/threadpool_interface.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/unbounded_work_queue.h"

namespace machina {
namespace data {

// An `UnboundedThreadPool` provides a mechanism for temporally multiplexing a
// potentially large number of "logical" threads onto a smaller number of
// "physical" threads. The multiplexing is achieved by using an
// `UnboundedWorkQueue`.
class UnboundedThreadPool : public thread::ThreadPoolInterface {
 public:
  UnboundedThreadPool(Env* env, const string& thread_name)
      : unbounded_work_queue_(env, thread_name) {}
  UnboundedThreadPool(Env* env, const string& thread_name,
                      const ThreadOptions& thread_options)
      : unbounded_work_queue_(env, thread_name, thread_options) {}
  ~UnboundedThreadPool() override = default;

  // Returns an implementation of `ThreadFactory` that can be used to create
  // logical threads in this pool.
  std::shared_ptr<ThreadFactory> get_thread_factory();

  void Schedule(std::function<void()> fn) override;
  int NumThreads() const override;
  int CurrentThreadId() const override;

 private:
  class LogicalThreadFactory;
  class LogicalThreadWrapper;

  void ScheduleOnWorkQueue(std::function<void()> fn,
                           std::shared_ptr<Notification> done);

  UnboundedWorkQueue unbounded_work_queue_;
};

}  // namespace data
}  // namespace machina

#endif  // MACHINA_CORE_DATA_UNBOUNDED_THREAD_POOL_H_
