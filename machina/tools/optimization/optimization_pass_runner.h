/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
#ifndef MACHINA_TOOLS_OPTIMIZATION_OPTIMIZATION_PASS_RUNNER_H_
#define MACHINA_TOOLS_OPTIMIZATION_OPTIMIZATION_PASS_RUNNER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "machina/core/common_runtime/device.h"
#include "machina/core/common_runtime/optimization_registry.h"
#include "machina/core/framework/device.h"
#include "machina/core/framework/types.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/protobuf/config.pb.h"

namespace machina {

// OptimizationPassRunner can be initialized, populated with devices, then run
// to test individual Tensorflow Optimization passes.
class OptimizationPassRunner {
 public:
  explicit OptimizationPassRunner() : jit_level_(OptimizerOptions::DEFAULT) {}

  // Increasing the Jit level will cause XLA to compile parts of the machina
  // graph that it is able to.
  absl::Status SetJitLevel(OptimizerOptions::GlobalJitLevel jit_level);

  absl::Status Run(absl::string_view pass_to_run, GraphDef input,
                   GraphDef* result);

  absl::Status AddCpus(int count) {
    return AddDevices(machina::DEVICE_CPU, count);
  }

  absl::Status AddGpus(int count) {
    return AddDevices(machina::DEVICE_GPU, count);
  }

 private:
  absl::Status AddDevices(absl::string_view type, int count);

  OptimizerOptions::GlobalJitLevel jit_level_;
  std::vector<std::unique_ptr<Device>> devices_;
};

}  // namespace machina

#endif  // MACHINA_TOOLS_OPTIMIZATION_OPTIMIZATION_PASS_RUNNER_H_
