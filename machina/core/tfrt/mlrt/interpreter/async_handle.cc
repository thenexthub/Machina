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
#include "machina/core/tfrt/mlrt/interpreter/async_handle.h"

#include <memory>
#include <utility>

#include "machina/xla/tsl/concurrency/async_value_ref.h"
#include "machina/xla/tsl/concurrency/chain.h"
#include "machina/core/tfrt/mlrt/interpreter/context.h"

namespace mlrt {

std::pair<AsyncHandle::Promise, AsyncHandle> AsyncHandle::Allocate(
    const ExecutionContext& current) {
  auto user_contexts = current.CopyUserContexts();

  auto new_context = std::make_unique<ExecutionContext>(
      &current.loaded_executable(), std::move(user_contexts),
      current.user_error_loggers());
  new_context->set_work_queue(current.work_queue());

  auto shared_state = tsl::MakeConstructedAsyncValueRef<tsl::Chain>();

  Promise promise(shared_state);
  AsyncHandle handle(std::move(new_context), std::move(shared_state));
  return {std::move(promise), std::move(handle)};
}

}  // namespace mlrt
