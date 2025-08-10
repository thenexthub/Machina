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
#ifndef MACHINA_C_EAGER_TFE_EXECUTOR_INTERNAL_H_
#define MACHINA_C_EAGER_TFE_EXECUTOR_INTERNAL_H_

#include <memory>

#include "machina/core/common_runtime/eager/eager_executor.h"

struct TFE_Executor {
  explicit TFE_Executor(bool async, bool enable_streaming_enqueue,
                        int in_flight_nodes_limit)
      : owned_executor(new machina::EagerExecutor(
            async, enable_streaming_enqueue, in_flight_nodes_limit)) {}

  explicit TFE_Executor(machina::EagerExecutor* executor)
      : owned_executor(nullptr), unowned_executor(executor) {}

  machina::EagerExecutor* executor() {
    return owned_executor == nullptr ? unowned_executor : owned_executor.get();
  }

  std::unique_ptr<machina::EagerExecutor> owned_executor;
  machina::EagerExecutor* unowned_executor;
};

#endif  // MACHINA_C_EAGER_TFE_EXECUTOR_INTERNAL_H_
