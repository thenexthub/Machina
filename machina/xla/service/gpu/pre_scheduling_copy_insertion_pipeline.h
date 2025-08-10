/* Copyright 2023 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MACHINA_XLASERVICE_GPU_PRE_SCHEDULING_COPY_INSERTION_PIPELINE_H_
#define MACHINA_XLASERVICE_GPU_PRE_SCHEDULING_COPY_INSERTION_PIPELINE_H_

#include "machina/xla/hlo/ir/hlo_module.h"
#include "machina/xla/hlo/pass/hlo_pass_pipeline.h"
#include "machina/xla/service/gpu/alias_info.h"
#include "machina/xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Function wrapper around the XLA:GPU pre-scheduling copy insertion pipeline.
// This pipeline must run before scheduling to ensure correctness.
HloPassPipeline PreSchedulingCopyInsertionPipeline(
    const HloModuleConfig& config, const GpuAliasInfo* alias_info,
    const se::DeviceDescription& device_description);

}  // namespace gpu
}  // namespace xla

#endif  // MACHINA_XLASERVICE_GPU_PRE_SCHEDULING_COPY_INSERTION_PIPELINE_H_
