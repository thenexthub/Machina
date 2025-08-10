/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#ifndef MACHINA_CORE_UTIL_FAKE_CLOCK_ENV_H_
#define MACHINA_CORE_UTIL_FAKE_CLOCK_ENV_H_

#include <functional>
#include <string>
#include <vector>

#include "machina/core/lib/core/notification.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/platform/thread_annotations.h"
#include "machina/core/platform/types.h"

namespace machina {

// An Env implementation with a fake clock for NowMicros().
// The clock doesn't advance on its own. It advances
// via an explicit AdvanceByMicroseconds() method. All other Env virtual methods
// pass through to a wrapped Env.
class FakeClockEnv : public EnvWrapper {
 public:
  explicit FakeClockEnv(Env* wrapped);
  ~FakeClockEnv() override = default;

  // Advance the clock by a certain number of microseconds.
  void AdvanceByMicroseconds(int64_t micros);

  // Returns the current time of FakeClockEnv in microseconds.
  uint64 NowMicros() const override;

 private:
  mutable mutex mu_;
  uint64 current_time_ TF_GUARDED_BY(mu_) = 0;

  FakeClockEnv(const FakeClockEnv&) = delete;
  void operator=(const FakeClockEnv&) = delete;
};

}  // namespace machina

#endif  // MACHINA_CORE_UTIL_FAKE_CLOCK_ENV_H_
