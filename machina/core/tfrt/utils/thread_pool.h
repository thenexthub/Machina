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
#ifndef MACHINA_CORE_TFRT_UTILS_THREAD_POOL_H_
#define MACHINA_CORE_TFRT_UTILS_THREAD_POOL_H_

#include <functional>
#include <string>
#include <utility>

#include "machina/core/platform/env.h"
#include "machina/core/platform/threadpool.h"
#include "machina/core/platform/threadpool_interface.h"

namespace machina {
namespace tfrt_stub {

class TfThreadPool : public thread::ThreadPoolInterface {
 public:
  explicit TfThreadPool(const std::string& name, int num_threads)
      : underlying_threadpool_(machina::Env::Default(), name, num_threads) {}

  void Schedule(std::function<void()> fn) override {
    underlying_threadpool_.Schedule(std::move(fn));
  }

  void ScheduleWithHint(std::function<void()> fn, int start, int end) override {
    underlying_threadpool_.ScheduleWithHint(std::move(fn), start, end);
  }

  void Cancel() override {
    underlying_threadpool_.AsEigenThreadPool()->Cancel();
  }

  int NumThreads() const override {
    return underlying_threadpool_.NumThreads();
  }

  int CurrentThreadId() const override {
    return underlying_threadpool_.CurrentThreadId();
  }

 private:
  machina::thread::ThreadPool underlying_threadpool_;
};

}  // namespace tfrt_stub
}  // namespace machina

#endif  // MACHINA_CORE_TFRT_UTILS_THREAD_POOL_H_
