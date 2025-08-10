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

#include "machina/compiler/mlir/tf2xla/mlir_bridge_rollout_policy.h"

#include <optional>

#include "machina/compiler/jit/flags.h"
#include "machina/core/framework/function.h"
#include "machina/core/graph/graph.h"
#include "machina/core/protobuf/config.pb.h"

namespace machina {

MlirBridgeRolloutPolicy GetMlirBridgeRolloutPolicy(
    const machina::Graph& graph,
    const FunctionLibraryDefinition* function_library,
    std::optional<ConfigProto> config_proto,
    bool is_supported_by_replicated_brige, bool is_v1_compat,
    bool record_stats) {
  switch (GetMlirBridgeRolloutState(config_proto)) {
    case ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_ENABLED:
      return MlirBridgeRolloutPolicy::kEnabledByUser;
    case ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_DISABLED:
      return MlirBridgeRolloutPolicy::kDisabledByUser;
    default:
      // User did not explicitly enable or disable the bridge. For now, disable
      // the bridge.
      return MlirBridgeRolloutPolicy::kDisabledAfterGraphAnalysis;
  }
}

void LogGraphFeatures(const Graph& graph,
                      const FunctionLibraryDefinition* function_library,
                      std::optional<ConfigProto> config_proto,
                      bool is_v1_compat) {}

}  // namespace machina
