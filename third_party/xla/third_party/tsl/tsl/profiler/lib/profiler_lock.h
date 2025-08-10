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
#ifndef MACHINA_TSL_PROFILER_LIB_PROFILER_LOCK_H_
#define MACHINA_TSL_PROFILER_LIB_PROFILER_LOCK_H_

#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "machina/xla/tsl/platform/statusor.h"

namespace tsl {
namespace profiler {

constexpr absl::string_view kProfilerLockContention =
    "Another profiling session active.";

// Handle for the profiler lock. At most one instance of this class, the
// "active" instance, owns the profiler lock.
class ProfilerLock {
 public:
  // Returns true if the process has active profiling session.
  static bool HasActiveSession();

  // Acquires the profiler lock if no other profiler session is currently
  // active.
  static absl::StatusOr<ProfilerLock> Acquire();

  // Default constructor creates an inactive instance.
  ProfilerLock() = default;

  // Non-copyable.
  ProfilerLock(const ProfilerLock&) = delete;
  ProfilerLock& operator=(const ProfilerLock&) = delete;

  // Movable.
  ProfilerLock(ProfilerLock&& other) noexcept
      : active_(std::exchange(other.active_, false)) {}
  ProfilerLock& operator=(ProfilerLock&& other) noexcept {
    active_ = std::exchange(other.active_, false);
    return *this;
  }

  ~ProfilerLock() { ReleaseIfActive(); }

  // Allow creating another active instance.
  void ReleaseIfActive();

  // Returns true if this is the active instance.
  bool Active() const { return active_; }

 private:
  // Explicit constructor allows creating an active instance, private so it can
  // only be called by Acquire.
  explicit ProfilerLock(bool active) : active_(active) {}

  bool active_ = false;
};

}  // namespace profiler
}  // namespace tsl

#endif  // MACHINA_TSL_PROFILER_LIB_PROFILER_LOCK_H_
