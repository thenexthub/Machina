/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#include "machina/core/kernels/batching_util/fake_clock_env.h"

#include <string>

namespace machina {
namespace serving {
namespace test_util {

FakeClockEnv::FakeClockEnv(Env* wrapped) : EnvWrapper(wrapped) {}

void FakeClockEnv::AdvanceByMicroseconds(int micros) {
  {
    mutex_lock l(mu_);
    current_time_ += micros;
    for (auto it = sleeping_threads_.begin(); it != sleeping_threads_.end();) {
      if (current_time_ >= it->wake_time) {
        it->wake_notification->Notify();
        it = sleeping_threads_.erase(it);
      } else {
        ++it;
      }
    }
  }
}

void FakeClockEnv::BlockUntilSleepingThread(uint64 wake_time) {
  for (;;) {
    {
      mutex_lock l(mu_);
      for (auto it = sleeping_threads_.begin(); it != sleeping_threads_.end();
           ++it) {
        if (it->wake_time == wake_time) {
          return;
        }
      }
    }
    EnvWrapper::SleepForMicroseconds(100);
  }
}

void FakeClockEnv::BlockUntilThreadsAsleep(int num_threads) {
  for (;;) {
    {
      mutex_lock l(mu_);
      if (num_threads <= sleeping_threads_.size()) {
        return;
      }
    }
    EnvWrapper::SleepForMicroseconds(100);
  }
}

uint64 FakeClockEnv::NowMicros() const {
  {
    mutex_lock l(mu_);
    return current_time_;
  }
}

void FakeClockEnv::SleepForMicroseconds(int64_t micros) {
  if (micros == 0) {
    return;
  }

  Notification wake_notification;
  {
    mutex_lock l(mu_);
    sleeping_threads_.push_back({current_time_ + micros, &wake_notification});
  }
  wake_notification.WaitForNotification();
}

}  // namespace test_util
}  // namespace serving
}  // namespace machina
