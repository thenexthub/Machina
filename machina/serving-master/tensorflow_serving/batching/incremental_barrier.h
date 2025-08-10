/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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

#ifndef MACHINA_CORE_KERNELS_BATCHING_UTIL_INCREMENTAL_BARRIER_H_
#define MACHINA_CORE_KERNELS_BATCHING_UTIL_INCREMENTAL_BARRIER_H_

#include <atomic>
#include <functional>

namespace machina {
namespace serving {

class InternalIncrementalBarrier;

// BarrierClosure (see
// https://github.com/chromium/chromium/blob/master/base/barrier_closure.h)
// executes a callback after it has been invoked |num_closures| times.
// Plus, `BarrierClosure` is a continuation-passing style abstraction and self-
// deleting.

// IncrementalBarrier is a convenience class to be used in place of a barrier
// closure, which is particularly helpful (e.g. simplify code) because callers
// don't need to calculate the |num_closures| beforehand.
//
// Example Usage:
//   void MakeCalls() {
//     typedef std::function<void()> Callback;
//     typedef std::function<void(Callback)> OtherCallback;
//     Callback done_callback = ...
//     OtherCallback cb1 = ...
//     OtherCallback cb2 = ...
//     std::thread threads[2];
//     {
//         IncrementalBarrier barrier(done_callback);
//         threads[0] = std::thread(cb1(barrier.Inc());
//         threads[1] = std::thread(cb2(barrier.Inc());
//         ... at this moment, `barrier` is incremented twice, and then
//         destructed....
//     }
//     threads[0].join();
//     threads[1].join();
//   }
//
//  `done_callback` will be called when both conditions are true:
//  1) after `barrier` is destructed.
//  2) Each `BarrierCallback` returned by `Inc` is called.
// This class is thread-safe.
class IncrementalBarrier {
 public:
  typedef std::function<void()> DoneCallback;
  typedef std::function<void()> BarrierCallback;
  explicit IncrementalBarrier(DoneCallback callback);

  ~IncrementalBarrier();

  // Returns a BarrierCallback (std::function) that individual task call to
  // signal its completeness.
  // The returned BarrierCallback outlives this `IncrementalBarrier` instance.
  // Furthermore, each task should eventually call the returned function, or
  // else done_callback wouldn't be called.
  BarrierCallback Inc();

 private:
  // self-deleting, thereby not owned by 'IncrementalBarrier'.
  InternalIncrementalBarrier* internal_barrier_;
};

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_BATCHING_UTIL_INCREMENTAL_BARRIER_H_
