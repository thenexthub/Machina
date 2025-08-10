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

#ifndef MACHINA_CORE_DISTRIBUTED_RUNTIME_RPC_EAGER_GRPC_EAGER_CLIENT_H_
#define MACHINA_CORE_DISTRIBUTED_RUNTIME_RPC_EAGER_GRPC_EAGER_CLIENT_H_

#include <memory>

#include "machina/core/distributed_runtime/eager/eager_client.h"
#include "machina/core/distributed_runtime/rpc/grpc_channel.h"

namespace machina {
namespace eager {
// The GrpcChannelCache is not owned.
EagerClientCache* NewGrpcEagerClientCache(
    std::shared_ptr<machina::GrpcChannelCache> channel);
}  // namespace eager
}  // namespace machina

#endif  // MACHINA_CORE_DISTRIBUTED_RUNTIME_RPC_EAGER_GRPC_EAGER_CLIENT_H_
