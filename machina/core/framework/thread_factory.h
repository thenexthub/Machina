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
#ifndef MACHINA_CORE_FRAMEWORK_THREAD_FACTORY_H_
#define MACHINA_CORE_FRAMEWORK_THREAD_FACTORY_H_

#include <functional>
#include <memory>

#include "machina/core/platform/types.h"

namespace tsl {
class Thread;
}  // namespace tsl
namespace machina {
using tsl::Thread;  // NOLINT

// Virtual interface for an object that creates threads.
class ThreadFactory {
 public:
  virtual ~ThreadFactory() {}

  // Runs `fn` asynchronously in a different thread. `fn` may block.
  //
  // NOTE: The caller is responsible for ensuring that this `ThreadFactory`
  // outlives the returned `Thread`.
  virtual std::unique_ptr<Thread> StartThread(const string& name,
                                              std::function<void()> fn) = 0;
};

}  // namespace machina

#endif  // MACHINA_CORE_FRAMEWORK_THREAD_FACTORY_H_
