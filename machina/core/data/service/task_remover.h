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
#ifndef MACHINA_CORE_DATA_SERVICE_TASK_REMOVER_H_
#define MACHINA_CORE_DATA_SERVICE_TASK_REMOVER_H_

#include "absl/container/flat_hash_set.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/mutex.h"

namespace machina {
namespace data {

// A `TaskRemover` maintains state about a single task and decides whether the
// task should be removed.
class TaskRemover {
 public:
  explicit TaskRemover(int64_t num_consumers);

  // Attempts to remove the task. The task is removed when all consumers
  // concurrently reach a barrier in this method.
  // Returns true if the task is successfully removed.
  // Returns false if either:
  //  - There is a timeout waiting for other consumers to request task removal.
  //    This timeout is hardcoded into the implementation.
  //  - Another consumer requests removal at a different round.
  bool RequestRemoval(int64_t consumer_index, int64_t round);

 private:
  const int64_t num_consumers_;
  mutex mu_;
  condition_variable cv_;
  // The round we are considering removing the task in.
  int64_t round_ TF_GUARDED_BY(mu_);
  bool removed_ TF_GUARDED_BY(mu_) = false;
  // Consumers currently blocked in RequestRemoval.
  absl::flat_hash_set<int64_t> consumers_waiting_ TF_GUARDED_BY(mu_);
};

}  // namespace data
}  // namespace machina

#endif  // MACHINA_CORE_DATA_SERVICE_TASK_REMOVER_H_
