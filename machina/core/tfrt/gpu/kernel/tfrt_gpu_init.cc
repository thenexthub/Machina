/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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
#include "machina/core/tfrt/gpu/kernel/tfrt_gpu_init.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "machina/xla/tsl/framework/serving_device_selector_policies.h"
#include "machina/core/common_runtime/gpu/gpu_serving_device_selector.h"
#include "machina/core/platform/status.h"
#include "machina/core/tfrt/gpu/kernel/gpu_runner.h"
#include "machina/core/tfrt/runtime/runtime.h"
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace machina {
namespace gpu {

absl::Status InitTfrtGpu(const GpuRunnerOptions& options,
                         machina::tfrt_stub::Runtime& runtime) {
  auto policy = std::make_unique<tsl::RoundRobinPolicy>();
  auto serving_device_selector =
      std::make_unique<machina::gpu::GpuServingDeviceSelector>(
          options.num_gpu_streams, std::move(policy));

  // We need to move `serving_device_selector` to the heap here, as
  // `AddCreateRuntimeResourceFn` requires a copyable callback.
  auto shared_serving_device_selector =
      std::shared_ptr<machina::gpu::GpuServingDeviceSelector>(
          serving_device_selector.release());
  runtime.AddCreateRuntimeResourceFn(
      [serving_device_selector = std::move(shared_serving_device_selector)](
          tfrt::ResourceContext* resource_ctx) mutable {
        resource_ctx->CreateResource<machina::gpu::GpuRunner>(
            machina::gpu::kGpuRunnerResourceName,
            serving_device_selector.get());
      });
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace machina
