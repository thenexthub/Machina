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

// Rewrites ConfigureDistributedTPU Op into a graph that configures each host.
//
// See the comment at the top of
// third_party/machina/core/ops/tpu_configuration_ops.cc to see the
// sequence of Ops used to configure a distributed TPU system.

#ifndef MACHINA_CORE_TPU_GRAPH_REWRITE_DISTRIBUTED_TPU_CONFIGURATION_REWRITE_PASS_H_
#define MACHINA_CORE_TPU_GRAPH_REWRITE_DISTRIBUTED_TPU_CONFIGURATION_REWRITE_PASS_H_

#include "absl/status/status.h"
#include "machina/core/common_runtime/optimization_registry.h"
#include "machina/core/platform/status.h"

namespace machina {

// Replaces dummy ConfigureDistributedTPU Ops assigned to TPU_SYSTEM
// devices with _ConfigureDistributedTPU and _WaitForDistributedTPU
// Ops on TPU_SYSTEM, and _InitializeHostForDistributedTPU on the CPU
// device of each host in the same job as the given TPU_SYSTEM device.
class DistributedTPUConfigurationRewritePass : public GraphOptimizationPass {
 public:
  absl::Status Run(const GraphOptimizationPassOptions& options) override;
};

// Replaces dummy ShutdownDistributedTPU Ops assigned to TPU_SYSTEM
// devices with _ShutdownDistributedTPU Ops on TPU_SYSTEM and
// _DisconnectHostFromDistributedTPUSystem on the CPU device of each
// host in the same job as the given TPU_SYSTEM device.
class DistributedTPUShutdownRewritePass : public GraphOptimizationPass {
 public:
  absl::Status Run(const GraphOptimizationPassOptions& options) override;
};

}  // namespace machina

#endif  // MACHINA_CORE_TPU_GRAPH_REWRITE_DISTRIBUTED_TPU_CONFIGURATION_REWRITE_PASS_H_
