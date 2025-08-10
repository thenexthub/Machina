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

#ifndef MACHINA_CORE_DISTRIBUTED_RUNTIME_PARTIAL_RUN_MGR_H_
#define MACHINA_CORE_DISTRIBUTED_RUNTIME_PARTIAL_RUN_MGR_H_

#include <unordered_map>

#include "machina/core/distributed_runtime/worker_interface.h"
#include "machina/core/framework/cancellation.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/platform/types.h"

namespace machina {

// PartialRunMgr keeps track of pending partial run requests, and ensures that
// the partial run is only marked complete when the corresponding executor is
// run to completion.
//
// In machina workers, the executor runs operations asynchronously until
// specified fetches (operations that return tensors) or targets (operations
// that don't return tensors) are reached. A PartialRun has two components: a
// setup which specifies all desired fetches and targets, and run calls that
// specify fetch values (from the setup calls) to retrieve.
// On the last partial run call, it is possible to satisfy the
// required fetches before the executor has completed running the graph to all
// the desired targets.
// PartialRunMgr is used to ensure that we don't complete and return the final
// partial run call to the user until both the partial run and executor have
// completed.
//
// PartialRunMgr is thread-safe.
class PartialRunMgr {
 public:
  // Find or create the CancellationManager associated with step_id.
  // The PartialRunMgr owns the cancellation_manager.
  // Returns true if a new CancellationManager was created
  // (i.e this is a new partial run).
  bool FindOrCreate(int step_id, CancellationManager** cancellation_manager);

  // Calls the final callback if the PartialRunRequest has already completed.
  // Otherwise stores the executor_status to be propagated when the
  // PartialRunRequest completes (PartialRunDone has been called).
  void ExecutorDone(int step_id, const absl::Status& executor_status);

  // Calls done if the executor has already completed (ExecutorDone has been
  // called). Otherwise, stores the status and done callback, calling them when
  // ExecutorDone is called. The callback will either be called by the calling
  // thread of either PartialRunDone or ExecutorDone.
  // If executor_status in ExecutorDone is not OK, it takes precedence over
  // status and is passed to the done callback.
  void PartialRunDone(int step_id, StatusCallback done,
                      const absl::Status& status);

 private:
  // PartialRunState stores state associated with a pending partial run request.
  // This is protected by the mutex in PartialRunMgr.
  struct PartialRunState {
    std::unique_ptr<CancellationManager> cancellation_manager;

    bool executor_done = false;
    StatusCallback final_callback = nullptr;
    absl::Status final_status;
  };

  mutex mu_;

  std::unordered_map<int, std::unique_ptr<PartialRunState>>
      step_id_to_partial_run_ TF_GUARDED_BY(mu_);
};

}  // namespace machina

#endif  // MACHINA_CORE_DISTRIBUTED_RUNTIME_PARTIAL_RUN_MGR_H_
