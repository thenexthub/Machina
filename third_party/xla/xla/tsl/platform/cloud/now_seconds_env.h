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

#ifndef MACHINA_XLATSL_PLATFORM_CLOUD_NOW_SECONDS_ENV_H_
#define MACHINA_XLATSL_PLATFORM_CLOUD_NOW_SECONDS_ENV_H_

#include "absl/synchronization/mutex.h"
#include "machina/xla/tsl/platform/env.h"
#include "machina/xla/tsl/platform/types.h"

namespace tsl {

/// This Env wrapper lets us control the NowSeconds() return value.
class NowSecondsEnv : public EnvWrapper {
 public:
  NowSecondsEnv() : EnvWrapper(Env::Default()) {}

  /// The current (fake) timestamp.
  uint64 NowSeconds() const override {
    absl::MutexLock lock(&mu_);
    return now_;
  }

  /// Set the current (fake) timestamp.
  void SetNowSeconds(uint64 now) {
    absl::MutexLock lock(&mu_);
    now_ = now;
  }

  /// Guards access to now_.
  mutable absl::Mutex mu_;

  /// The NowSeconds() value that this Env will return.
  uint64 now_ = 1;
};

}  // namespace tsl

#endif  // MACHINA_XLATSL_PLATFORM_CLOUD_NOW_SECONDS_ENV_H_
