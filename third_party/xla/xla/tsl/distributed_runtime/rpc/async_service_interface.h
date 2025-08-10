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

#ifndef MACHINA_XLATSL_DISTRIBUTED_RUNTIME_RPC_ASYNC_SERVICE_INTERFACE_H_
#define MACHINA_XLATSL_DISTRIBUTED_RUNTIME_RPC_ASYNC_SERVICE_INTERFACE_H_

namespace tsl {

// Represents an abstract asynchronous service that handles incoming
// RPCs with a polling loop.
class AsyncServiceInterface {
 public:
  virtual ~AsyncServiceInterface() {}

  // A blocking method that should be called to handle incoming RPCs.
  // This method will block until the service shuts down.
  virtual void HandleRPCsLoop() = 0;

  // Starts shutting down this service.
  //
  // NOTE(mrry): To shut down this service completely, the caller must
  // also shut down any servers that might share ownership of this
  // service's resources (e.g. completion queues).
  virtual void Shutdown() = 0;
};

}  // namespace tsl

#endif  // MACHINA_XLATSL_DISTRIBUTED_RUNTIME_RPC_ASYNC_SERVICE_INTERFACE_H_
