/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#include "machina/compiler/mlir/tfrt/utils/host_context.h"

#include <cstdint>
#include <memory>

#include "absl/log/log.h"
#include "machina/core/platform/logging.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime

namespace machina {

using ::tfrt::HostContext;

const char* const kDefaultHostDeviceName =
    "/job:localhost/replica:0/task:0/device:CPU:0";

std::unique_ptr<HostContext> CreateSingleThreadedHostContext() {
  return std::make_unique<HostContext>(
      [](const tfrt::DecodedDiagnostic& diag) {
        LOG(FATAL) << "Runtime error: " << diag.message() << "\n";
      },
      tfrt::CreateMallocAllocator(), tfrt::CreateSingleThreadedWorkQueue(),
      kDefaultHostDeviceName);
}

std::unique_ptr<HostContext> CreateMultiThreadedHostContext(
    int64_t num_threads) {
  return std::make_unique<HostContext>(
      [](const tfrt::DecodedDiagnostic& diag) {
        LOG(FATAL) << "Runtime error: " << diag.message() << "\n";
      },
      tfrt::CreateMallocAllocator(),
      tfrt::CreateMultiThreadedWorkQueue(num_threads,
                                         /*num_blocking_threads=*/1),
      kDefaultHostDeviceName);
}

}  // namespace machina
