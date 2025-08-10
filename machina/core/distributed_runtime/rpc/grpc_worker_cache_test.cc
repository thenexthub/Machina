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

#include "machina/core/distributed_runtime/rpc/grpc_worker_cache.h"

#include "machina/c/tf_status.h"
#include "machina/core/distributed_runtime/rpc/grpc_channel.h"
#include "machina/core/distributed_runtime/test_utils.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/strcat.h"
#include "machina/core/platform/test.h"
#include "machina/core/platform/threadpool.h"

namespace machina {

TEST(GrpcWorkerCacheTest, NewGrpcWorkerCache) {
  GrpcChannelSpec spec;
  TF_ASSERT_OK(
      spec.AddHostPortsJob("worker", {{0, "a:0"}, {1, "b:1"}, {2, "c:2"}}));
  ChannelCreationFunction channel_func =
      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
  auto channel_cache = std::shared_ptr<GrpcChannelCache>(
      NewGrpcChannelCache(spec, channel_func));
  std::unique_ptr<GrpcWorkerEnv> grpc_worker_env(CreateGrpcWorkerEnv());

  // We created a job with 3 tasks. Getting the task 0, 1, 2 should return valid
  // worker interfaces, and getting other tasks should return nullptr.
  std::unique_ptr<WorkerCacheInterface> worker_cache(
      NewGrpcWorkerCache(channel_cache, grpc_worker_env.get()));
  WorkerInterface* wi;
  wi = worker_cache->GetOrCreateWorker("/job:worker/replica:0/task:0");
  EXPECT_NE(wi, nullptr);
  worker_cache->ReleaseWorker("/job:worker/replica:0/task:0", wi);
  wi = worker_cache->GetOrCreateWorker("/job:worker/replica:0/task:1");
  EXPECT_NE(wi, nullptr);
  worker_cache->ReleaseWorker("/job:worker/replica:0/task:1", wi);
  wi = worker_cache->GetOrCreateWorker("/job:worker/replica:0/task:2");
  EXPECT_NE(wi, nullptr);
  worker_cache->ReleaseWorker("/job:worker/replica:0/task:2", wi);
  wi = worker_cache->GetOrCreateWorker("/job:worker/replica:0/task:3");
  EXPECT_EQ(wi, nullptr);

  // Test creating a worker cache instance with local worker, and getting the
  // worker instance with the specified local target.
  std::unique_ptr<TestWorkerInterface> local_wi;
  worker_cache.reset(NewGrpcWorkerCacheWithLocalWorker(
      channel_cache, grpc_worker_env.get(), local_wi.get(), "local_target"));
  wi = worker_cache->GetOrCreateWorker("local_target");
  EXPECT_EQ(wi, local_wi.get());
}

TEST(GrpcWorkerCacheTest, DestructWorkerCacheInThreadPool) {
  GrpcChannelSpec spec;
  TF_ASSERT_OK(
      spec.AddHostPortsJob("worker", {{0, "a:0"}, {1, "b:1"}, {2, "c:2"}}));
  ChannelCreationFunction channel_func =
      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
  auto channel_cache = std::shared_ptr<GrpcChannelCache>(
      NewGrpcChannelCache(spec, channel_func));
  std::unique_ptr<GrpcWorkerEnv> grpc_worker_env(CreateGrpcWorkerEnv());

  // The GrpcWorkerEnv threadpool is used for worker interfaces for gRPC
  // completion queue callbacks. Test worker cache destruction inside the
  // callbacks that runs in the GrpcWorkerEnv threadpool.
  WorkerCacheInterface* worker_cache =
      NewGrpcWorkerCache(channel_cache, grpc_worker_env.get());
  thread::ThreadPool* tp = grpc_worker_env->GetThreadPool();
  Notification n;
  tp->Schedule([worker_cache, &n] {
    delete worker_cache;
    n.Notify();
  });
  n.WaitForNotification();
}

}  // namespace machina
