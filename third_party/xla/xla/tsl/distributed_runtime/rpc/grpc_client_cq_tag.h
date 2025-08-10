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

#ifndef MACHINA_XLATSL_DISTRIBUTED_RUNTIME_RPC_GRPC_CLIENT_CQ_TAG_H_
#define MACHINA_XLATSL_DISTRIBUTED_RUNTIME_RPC_GRPC_CLIENT_CQ_TAG_H_

#include "machina/xla/tsl/platform/macros.h"

namespace tsl {

// Represents a pending asynchronous client call as a tag that can be
// stored in a `grpc::CompletionQueue`.
class GrpcClientCQTag {
 public:
  GrpcClientCQTag() = default;
  virtual ~GrpcClientCQTag() = default;

  // OnCompleted is invoked when the RPC has finished.
  // Implementations of OnCompleted can delete *this.
  virtual void OnCompleted(bool ok) = 0;

 private:
  GrpcClientCQTag(const GrpcClientCQTag&) = delete;
  void operator=(const GrpcClientCQTag&) = delete;
};

}  // namespace tsl

#endif  // MACHINA_XLATSL_DISTRIBUTED_RUNTIME_RPC_GRPC_CLIENT_CQ_TAG_H_
