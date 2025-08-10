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

#include "machina/core/graph/regularization/simple_delete.h"

#include <cstdint>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/strip.h"
#include "machina/core/framework/attr_value.pb.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/framework/versions.pb.h"
#include "machina/core/graph/regularization/util.h"
#include "machina/core/grappler/op_types.h"

namespace machina::graph_regularization {

namespace {

// This function mutates `graph_def`, changing the names and config_proto's
// of the Function nodes.
void RegularizeNodes(GraphDef* graph_def) {
  for (NodeDef& node : *graph_def->mutable_node()) {
    // Check if this is a function call.
    if (grappler::IsPartitionedCall(node) ||
        grappler::IsStatefulPartitionedCall(node)) {
      // Regularize "f" attribute, the function name for PartitionedCall and
      // and StatefulPartitionedCall ops, by stripping the suffix UID if it
      // has one.
      std::string function_name = node.attr().find("f")->second.func().name();
      absl::StatusOr<int64_t> uid = GetSuffixUID(function_name);
      if (uid.ok()) {
        node.mutable_attr()->find("f")->second.mutable_func()->set_name(
            std::string(
                absl::StripSuffix(function_name, std::to_string(*uid))));
      }
      // Erase the "config_proto" attribute which contains device-specific
      // information.
      auto node_config_proto = node.mutable_attr()->find("config_proto");
      if (node_config_proto != node.attr().end()) {
        node_config_proto->second.mutable_s()->erase();
      }
    }
    // Erase the value of string constants, which can vary based on platform.
    if (grappler::IsConstant(node)) {
      if (node.attr().at("dtype").type() == DT_STRING) {
        node.mutable_attr()->find("value")->second.clear_value();
      }
    }
  }
}
}  // namespace

void SimpleDelete(GraphDef& graph_def) {
  // The GraphDef contains two main sections: a list of nodes and the
  // FunctionDefLibrary. Regularization treats these two sections separately.
  RegularizeNodes(&graph_def);
  // TODO(b/240173815): Complete canonicalization of the FunctionDefLibrary.
  // For now, we just completely clear the FunctionDefLibrary.
  graph_def.mutable_library()->Clear();
  graph_def.mutable_versions()->Clear();
}

}  // namespace machina::graph_regularization
