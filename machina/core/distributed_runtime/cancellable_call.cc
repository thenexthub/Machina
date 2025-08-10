/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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
#include "machina/core/distributed_runtime/cancellable_call.h"

namespace machina {

void CancellableCall::Start(const StatusCallback& done) {
  if (cancel_mgr_ == nullptr) {
    IssueCall(done);
    return;
  }
  CancellationToken token = cancel_mgr_->get_cancellation_token();
  const bool not_yet_cancelled =
      cancel_mgr_->RegisterCallback(token, [this]() { Cancel(); });
  if (not_yet_cancelled) {
    IssueCall([this, token, done](const absl::Status& s) {
      cancel_mgr_->DeregisterCallback(token);
      done(s);
    });
  } else {
    done(errors::Cancelled("RPC Request was cancelled"));
  }
}

void CancellableCall::Cancel() {
  {
    mutex_lock l(mu_);
    if (is_cancelled_) {
      return;
    }
    is_cancelled_ = true;
  }
  opts_.StartCancel();
}

}  // namespace machina
