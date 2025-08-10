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
#include "machina/core/tfrt/runtime/runtime.h"

#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "machina/core/framework/types.h"
#include "machina/core/platform/cpu_info.h"
#include "machina/core/platform/status.h"
#include "machina/core/runtime_fallback/kernel/kernel_fallback_tensor.h"
#include "tfrt/cpu/core_runtime/cpu_op_handler.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/tensor/scalar_host_tensor.h"  // from @tf_runtime

constexpr char const kDefaultHostDeviceName[] =
    "/job:localhost/replica:0/task:0/device:CPU:0";

namespace machina {
namespace tfrt_stub {
namespace {
Runtime** GetGlobalRuntimeInternal() {
  static Runtime* tfrt_runtime = nullptr;
  return &tfrt_runtime;
}
}  // namespace

std::unique_ptr<Runtime> Runtime::Create(
    std::unique_ptr<WorkQueueInterface> work_queue) {
  auto* work_queue_ptr = work_queue.get();
  auto expected_core_runtime = tfrt::CoreRuntime::Create(
      [](const tfrt::DecodedDiagnostic& diag) { LOG(ERROR) << diag.message(); },
      tfrt::CreateMallocAllocator(), std::move(work_queue),
      kDefaultHostDeviceName);
  DCHECK(expected_core_runtime);

  // We don't use std::make_unique here because the constructor should better be
  // private.
  return std::unique_ptr<Runtime>(
      new Runtime(std::move(expected_core_runtime.get()), work_queue_ptr));
}

std::unique_ptr<Runtime> Runtime::Create(int num_inter_op_threads,
                                         int num_intra_op_threads) {
  if (num_intra_op_threads <= 0)
    num_intra_op_threads = machina::port::MaxParallelism();
  return Runtime::Create(
      WrapDefaultWorkQueue(tfrt::CreateMultiThreadedWorkQueue(
          num_intra_op_threads, num_inter_op_threads)));
}

Runtime::Runtime(std::unique_ptr<tfrt::CoreRuntime> core_runtime,
                 WorkQueueInterface* work_queue)
    : core_runtime_(std::move(core_runtime)), work_queue_(work_queue) {
  DCHECK(work_queue_);
}

Runtime::~Runtime() = default;

Runtime* GetGlobalRuntime() { return *GetGlobalRuntimeInternal(); }

void SetGlobalRuntime(std::unique_ptr<Runtime> runtime) {
  Runtime** global_runtime_ptr = GetGlobalRuntimeInternal();
  if (*global_runtime_ptr) {
    delete *global_runtime_ptr;
  }
  *global_runtime_ptr = runtime.release();
}
}  // namespace tfrt_stub
}  // namespace machina
