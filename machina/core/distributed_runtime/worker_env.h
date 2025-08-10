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

#ifndef MACHINA_CORE_DISTRIBUTED_RUNTIME_WORKER_ENV_H_
#define MACHINA_CORE_DISTRIBUTED_RUNTIME_WORKER_ENV_H_

#include <vector>

#include "machina/core/platform/types.h"

namespace tsl {
class Env;
namespace thread {
class ThreadPool;
}  // namespace thread
}  // namespace tsl
namespace machina {
using Env = tsl::Env;

namespace thread {
using tsl::thread::ThreadPool;
}  // namespace thread

class CollectiveExecutorMgrInterface;
class Device;
class DeviceMgr;
class RendezvousMgrInterface;
class SessionMgr;

// The worker environment class, which holds a bag of pointers to
// per-worker singletons.
//
// WorkerEnv does not own its member pointers.
struct WorkerEnv {
  Env* env = nullptr;

  // session_mgr encapsulates state for each session.
  SessionMgr* session_mgr = nullptr;

  // In large scaled distributed training, many singleton components (e.g.
  // Rendezvous) can becomes the bottleneck of the system. This field allows
  // us to shard the single components. This number will scale up with number
  // of tasks in this cluster. It is always greater than 1.
  int experimental_num_shards = 1;

  // device_mgr manages local devices (cpu and gpu). The WorkerService
  // is the network interface for managed devices.
  //
  // Note: Please use the device_mgr associated with your session if appropriate
  // instead of this one. Using this device_mgr does not support ClusterSpec
  // propagated sessions.
  DeviceMgr* device_mgr = nullptr;

  // A set of rendezvous keyed by step ids.
  RendezvousMgrInterface* rendezvous_mgr = nullptr;

  // Generates per-step CollectiveExecutors and has access to utilities
  // supporting collective operations.
  std::unique_ptr<CollectiveExecutorMgrInterface> collective_executor_mgr;

  // A pool of threads for scheduling compute work.
  thread::ThreadPool* compute_pool = nullptr;
};

}  // end namespace machina

#endif  // MACHINA_CORE_DISTRIBUTED_RUNTIME_WORKER_ENV_H_
