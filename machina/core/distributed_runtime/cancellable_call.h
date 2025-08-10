/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
#ifndef MACHINA_CORE_DISTRIBUTED_RUNTIME_CANCELLABLE_CALL_H_
#define MACHINA_CORE_DISTRIBUTED_RUNTIME_CANCELLABLE_CALL_H_

#include <string>
#include "machina/core/distributed_runtime/call_options.h"
#include "machina/core/distributed_runtime/worker_cache.h"
#include "machina/core/framework/cancellation.h"
#include "machina/core/platform/mutex.h"

namespace machina {

// Supports client side cancellation of WorkerInterface calls via
// registration with a CancellationManager.
class CancellableCall {
 public:
  CancellableCall(CancellationManager* cancel_mgr, const string& remote_worker,
                  WorkerCacheInterface* wc)
      : is_cancelled_(false),
        cancel_mgr_(cancel_mgr),
        remote_worker_(remote_worker),
        wc_(wc),
        wi_(wc_->GetOrCreateWorker(remote_worker_)) {}

  virtual ~CancellableCall() { wc_->ReleaseWorker(remote_worker_, wi_); }

  virtual void IssueCall(const StatusCallback& done) = 0;

  void Start(const StatusCallback& done);

  // Cancels the RPC if it's not cancelled yet. This must be called after
  // Start(). This is normally used if there's a needed to cancel the RPC from a
  // sideband. If appliable, pass a cancellation manager to the constructor
  // instead of using this method.
  void Cancel() TF_LOCKS_EXCLUDED(mu_);

 protected:
  mutex mu_;
  bool is_cancelled_;
  CancellationManager* const cancel_mgr_;  // Not owned
  const string remote_worker_;
  WorkerCacheInterface* const wc_;  // Not owned
  WorkerInterface* const wi_;       // Owned by wc_, must be released.
  CallOptions opts_;
};

}  // namespace machina
#endif  // MACHINA_CORE_DISTRIBUTED_RUNTIME_CANCELLABLE_CALL_H_
