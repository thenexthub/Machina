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

#ifndef MACHINA_CORE_DISTRIBUTED_RUNTIME_EAGER_DESTROY_TENSOR_HANDLE_NODE_H_
#define MACHINA_CORE_DISTRIBUTED_RUNTIME_EAGER_DESTROY_TENSOR_HANDLE_NODE_H_

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "machina/core/common_runtime/eager/eager_executor.h"
#include "machina/core/distributed_runtime/eager/eager_client.h"
#include "machina/core/protobuf/eager_service.pb.h"

namespace machina {
namespace eager {

// DestroyTensorHandleNode is an implementation of EagerNode which enqueues a
// request to destroy a remote tensor handle.
class DestroyTensorHandleNode : public machina::AsyncEagerNode {
 public:
  DestroyTensorHandleNode(std::unique_ptr<EnqueueRequest> request,
                          core::RefCountPtr<EagerClient> eager_client,
                          bool ready)
      : machina::AsyncEagerNode(),
        request_(std::move(request)),
        eager_client_(std::move(eager_client)),
        ready_(ready) {}

  ~DestroyTensorHandleNode() override {}

  void RunAsync(StatusCallback done) override {
    EnqueueResponse* response = new EnqueueResponse;
    bool ready = ready_;
    // NOTE(fishx): Don't use StreamingEnqueueAsync here. When a
    // StreamingEnqueueAsync request fails all following requests will fail as
    // well. We don't want this request poison following requests since it is
    // safe to ignore a failing destroy tensor handle request.
    eager_client_->EnqueueAsync(
        /*call_opts=*/nullptr, request_.get(), response,
        [response, ready, done](const absl::Status& s) {
          // Omit the warning if:
          // 1. The remote tensor isn't ready.
          // 2. Lost connection to remote worker. In this case client will
          //    crash. We don't want to spam user with redundant warning logs.
          if (!s.ok() && ready && !absl::IsUnavailable(s)) {
            LOG_EVERY_N_SEC(WARNING, 60)
                << "Ignoring an error encountered when deleting "
                   "remote tensors handles: "
                << s.ToString();
          }
          done(absl::OkStatus());
          delete response;
        });
  }

  void Abort(absl::Status status) override {}

  // Remote node deletions are best effort
  bool Fatal() const override { return false; }

  string DebugString() const override {
    string out = "[DestroyTensorHandleNode]";
    strings::StrAppend(&out, " request: ", request_->DebugString());
    return out;
  }

 private:
  std::unique_ptr<EnqueueRequest> request_;
  core::RefCountPtr<EagerClient> eager_client_;
  const string remote_task_;
  bool ready_;
};

}  // namespace eager
}  // namespace machina

#endif  // MACHINA_CORE_DISTRIBUTED_RUNTIME_EAGER_DESTROY_TENSOR_HANDLE_NODE_H_
