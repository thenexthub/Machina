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

#include "machina/core/common_runtime/function_optimization_registry.h"

#include <string>

#include "machina/core/framework/metrics.h"

namespace machina {

void FunctionOptimizationPassRegistry::Init(
    std::unique_ptr<FunctionOptimizationPass> pass) {
  DCHECK(!pass_) << "Only one pass should be set.";
  pass_ = std::move(pass);
}

absl::Status FunctionOptimizationPassRegistry::Run(
    const std::string& function_name, const DeviceSet& device_set,
    const ConfigProto& config_proto,
    const FunctionOptimizationPass::FunctionOptions& function_options,
    std::unique_ptr<Graph>* graph, FunctionLibraryDefinition* flib_def,
    std::vector<std::string>* control_ret_node_names,
    bool* control_rets_updated) {
  if (!pass_) return absl::OkStatus();

  machina::metrics::ScopedCounter<2> timings(
      machina::metrics::GetGraphOptimizationCounter(),
      {"GraphOptimizationPass", "FunctionOptimizationPassRegistry"});

  return pass_->Run(function_name, device_set, config_proto, function_options,
                    graph, flib_def, control_ret_node_names,
                    control_rets_updated);
}

// static
FunctionOptimizationPassRegistry& FunctionOptimizationPassRegistry::Global() {
  static FunctionOptimizationPassRegistry* kGlobalRegistry =
      new FunctionOptimizationPassRegistry;
  return *kGlobalRegistry;
}

}  // namespace machina
