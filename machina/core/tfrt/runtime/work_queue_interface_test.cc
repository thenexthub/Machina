/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#include "machina/core/tfrt/runtime/work_queue_interface.h"

#include <thread>
#include <utility>

#include <gtest/gtest.h>
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/tfrt/utils/thread_pool.h"
#include "tfrt/cpp_tests/test_util.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/task_function.h"  // from @tf_runtime

namespace machina {
namespace tfrt_stub {
namespace {

TEST(DefaultWorkQueueWrapperTest, Name) {
  auto work_queue = tfrt::CreateSingleThreadedWorkQueue();
  auto work_queue_ptr = work_queue.get();
  auto work_queue_wrapper = WrapDefaultWorkQueue(std::move(work_queue));

  EXPECT_EQ(work_queue_wrapper->name(), work_queue_ptr->name());
}

TEST(DefaultWorkQueueWrapperTest, AddTask_OnlyTask) {
  auto work_queue = tfrt::CreateSingleThreadedWorkQueue();
  auto work_queue_wrapper = WrapDefaultWorkQueue(std::move(work_queue));

  auto av = tfrt::MakeUnconstructedAsyncValueRef<int>().ReleaseRCRef();
  work_queue_wrapper->AddTask(
      tfrt::TaskFunction([av] { av->emplace<int>(0); }));
  work_queue_wrapper->Await(std::move(av));
}

TEST(DefaultWorkQueueWrapperTest, AddBlockingTask_TaskAndAllowQueueing) {
  auto work_queue = tfrt::CreateSingleThreadedWorkQueue();
  auto work_queue_wrapper = WrapDefaultWorkQueue(std::move(work_queue));

  auto av = tfrt::MakeUnconstructedAsyncValueRef<int>().ReleaseRCRef();
  std::thread thread{[&] {
    auto work = work_queue_wrapper->AddBlockingTask(
        tfrt::TaskFunction([&] { av->emplace<int>(0); }),
        /*allow_queuing=*/true);
  }};
  work_queue_wrapper->Await(std::move(av));
  thread.join();
}

TEST(DefaultWorkQueueWrapperTest, GetParallelismLevel) {
  auto work_queue = tfrt::CreateSingleThreadedWorkQueue();
  auto work_queue_ptr = work_queue.get();
  auto work_queue_wrapper = WrapDefaultWorkQueue(std::move(work_queue));

  EXPECT_EQ(work_queue_wrapper->GetParallelismLevel(),
            work_queue_ptr->GetParallelismLevel());
}

TEST(DefaultWorkQueueWrapperTest, IsInWorkerThread) {
  auto work_queue = tfrt::CreateSingleThreadedWorkQueue();
  auto work_queue_ptr = work_queue.get();
  auto work_queue_wrapper = WrapDefaultWorkQueue(std::move(work_queue));

  EXPECT_EQ(work_queue_wrapper->IsInWorkerThread(),
            work_queue_ptr->IsInWorkerThread());
}

TEST(DefaultWorkQueueWrapperTest, IntraOpThreadPool) {
  auto work_queue = tfrt::CreateSingleThreadedWorkQueue();
  TfThreadPool intra_op_thread_pool(/*name=*/"tf_intra",
                                    /*num_threads=*/1);
  auto work_queue_wrapper =
      WrapDefaultWorkQueue(std::move(work_queue), &intra_op_thread_pool);

  TF_ASSERT_OK_AND_ASSIGN(auto queue, work_queue_wrapper->InitializeRequest(
                                          /*request_id=*/0));
  EXPECT_NE(queue, nullptr);
  EXPECT_EQ(queue->GetIntraOpThreadPool(), &intra_op_thread_pool);
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace machina
