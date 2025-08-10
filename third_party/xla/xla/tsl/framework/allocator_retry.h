/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#ifndef MACHINA_MACHINA_XLA_TSL_FRAMEWORK_ALLOCATOR_RETRY_H_
#define MACHINA_MACHINA_XLA_TSL_FRAMEWORK_ALLOCATOR_RETRY_H_

#include <cstddef>
#include <functional>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "machina/xla/tsl/platform/env.h"

namespace tsl {

// A retrying wrapper for a memory allocator.
class AllocatorRetry {
 public:
  AllocatorRetry();
  ~AllocatorRetry();

  // Call 'alloc_func' to obtain memory.  On first call,
  // 'verbose_failure' will be false.  If return value is nullptr,
  // then wait up to 'max_millis_to_wait' milliseconds, retrying each
  // time a call to DeallocateRaw() is detected, until either a good
  // pointer is returned or the deadline is exhausted.  If the
  // deadline is exhausted, try one more time with 'verbose_failure'
  // set to true.  The value returned is either the first good pointer
  // obtained from 'alloc_func' or nullptr.
  void* AllocateRaw(std::function<void*(size_t alignment, size_t num_bytes,
                                        bool verbose_failure)>
                        alloc_func,
                    int max_millis_to_wait, size_t alignment, size_t bytes);

  // Called to notify clients that some memory was returned.
  void NotifyDealloc();

 private:
  Env* env_;
  absl::Mutex mu_;
  absl::CondVar memory_returned_ ABSL_GUARDED_BY(mu_);
};

// Implementation details below
inline void AllocatorRetry::NotifyDealloc() {
  absl::MutexLock l(&mu_);
  memory_returned_.SignalAll();
}

}  // namespace tsl

#endif  // MACHINA_MACHINA_XLA_TSL_FRAMEWORK_ALLOCATOR_RETRY_H_
