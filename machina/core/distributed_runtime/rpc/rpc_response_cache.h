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
#ifndef MACHINA_CORE_DISTRIBUTED_RUNTIME_RPC_RPC_RESPONSE_CACHE_H_
#define MACHINA_CORE_DISTRIBUTED_RUNTIME_RPC_RPC_RESPONSE_CACHE_H_

#include <memory>
#include <unordered_map>
#include <vector>

#include "machina/core/framework/tensor.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/lib/gtl/flatmap.h"
#include "machina/core/platform/mutex.h"

// gRPC response caching.  Most WorkerService methods cannot be retried directly
// as they will fail or deadlock.  To enable retrying, we can instead cache
// responses and reply to duplicate requests from the cache. The cache will be
// cleaned when the MarkRecvFinishedRequest is received from the receiver or the
// session step is completed.
namespace machina {

// Track and cache the state of worker service RPCs.  An RPC can be in 3 states:
//
// * PENDING: this is the first call of the RPC, and it will transition to
// * ACTIVE: another thread is active processing this RPC
// * FINISHED: the worker has finished processing the method

class RpcResponseCache {
 public:
  using FinishResponseCB = std::function<void(
      const Tensor& tensor, bool is_dead, const absl::Status& status)>;

  // Add the given request to the cache.
  // If the request is in the cache,
  //    If it is finished, invoke `cb` immediately
  //    If active, cb will be invoked when the current call completes.
  //    In either case, return true.
  // Otherwise, store the request and cb in the cache, and return false.
  // Note FinishResponseCB is assumed to be thread-safe.
  bool QueueRequest(int64_t request_id, int64_t step_id,
                    const FinishResponseCB& cb);

  // Fill the response cache for the given request_id and respond to all
  // pending request.
  void RequestFinished(int64_t request_id, const Tensor& tensor, bool is_dead,
                       const absl::Status& status);

  // Erase the cache entry with the given request_id
  void EraseRequestId(int64_t request_id);

  // Erase cache entries with the given step_id
  void CleanEntriesForStep(int64_t step_id);

  int64_t size();

 private:
  struct ResponseCacheEntry {
    enum class State {
      PENDING = 0,
      ACTIVE = 1,
      FINISHED = 2,
    };

    State state = State::PENDING;
    int64_t step_id = -1;
    Tensor tensor;
    bool is_dead = false;
    absl::Status response_status;

    void FinishResponse(const FinishResponseCB& cb) const {
      cb(tensor, is_dead, response_status);
    }
    std::vector<FinishResponseCB> callbacks;
  };

  mutex mu_;
  // response_cache_ is expected to be small, as entries are cleared immediately
  // on ack from the receiver.
  gtl::FlatMap<int64_t, ResponseCacheEntry> response_cache_ TF_GUARDED_BY(mu_);
};

}  // namespace machina

#endif  // MACHINA_CORE_DISTRIBUTED_RUNTIME_RPC_RPC_RESPONSE_CACHE_H_
