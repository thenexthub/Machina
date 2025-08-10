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

#include "machina/core/kernels/batching_util/basic_batch_scheduler.h"

#include <utility>

#include "machina/core/kernels/batching_util/batch_scheduler.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace serving {
namespace {

class FakeTask : public BatchTask {
 public:
  explicit FakeTask(size_t size) : size_(size) {}

  ~FakeTask() override = default;

  size_t size() const override { return size_; }

 private:
  const size_t size_;

  FakeTask(const FakeTask&) = delete;
  void operator=(const FakeTask&) = delete;
};

// Creates a FakeTask of size 'task_size', and calls 'scheduler->Schedule()'
// on that task. Returns the resulting status.
absl::Status ScheduleTask(size_t task_size,
                          BatchScheduler<FakeTask>* scheduler) {
  std::unique_ptr<FakeTask> task(new FakeTask(task_size));
  absl::Status status = scheduler->Schedule(&task);
  // Schedule() should have consumed 'task' iff it returned Status::OK.
  CHECK_EQ(status.ok(), task == nullptr);
  return status;
}

// Since BasicBatchScheduler is implemented as a thin wrapper around
// SharedBatchScheduler, we only do some basic testing. More comprehensive
// testing is done in shared_batch_scheduler_test.cc.

TEST(BasicBatchSchedulerTest, Basic) {
  bool callback_called = false;
  auto callback = [&callback_called](std::unique_ptr<Batch<FakeTask>> batch) {
    callback_called = true;
    ASSERT_TRUE(batch->IsClosed());
    ASSERT_EQ(2, batch->num_tasks());
    EXPECT_EQ(3, batch->task(0).size());
    EXPECT_EQ(5, batch->task(1).size());
  };
  {
    BasicBatchScheduler<FakeTask>::Options options;
    options.max_batch_size = 10;
    options.batch_timeout_micros = 100 * 1000;  // 100 milliseconds
    options.num_batch_threads = 1;
    options.max_enqueued_batches = 3;
    std::unique_ptr<BasicBatchScheduler<FakeTask>> scheduler;
    TF_ASSERT_OK(
        BasicBatchScheduler<FakeTask>::Create(options, callback, &scheduler));
    EXPECT_EQ(10, scheduler->max_task_size());
    EXPECT_EQ(0, scheduler->NumEnqueuedTasks());
    EXPECT_EQ(3 * 10, scheduler->SchedulingCapacity());
    TF_ASSERT_OK(ScheduleTask(3, scheduler.get()));
    EXPECT_EQ(1, scheduler->NumEnqueuedTasks());
    EXPECT_EQ((3 * 10) - 3, scheduler->SchedulingCapacity());
    TF_ASSERT_OK(ScheduleTask(5, scheduler.get()));
    EXPECT_EQ(2, scheduler->NumEnqueuedTasks());
    EXPECT_EQ((3 * 10) - (3 + 5), scheduler->SchedulingCapacity());
  }
  EXPECT_TRUE(callback_called);
}

}  // namespace
}  // namespace serving
}  // namespace machina
