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

#include "machina/core/debug/debugger_state_impl.h"

#include "machina/core/debug/debug_graph_utils.h"
#include "machina/core/debug/debug_io_utils.h"

namespace machina {

DebuggerState::DebuggerState(const DebugOptions& debug_options) {
  for (const DebugTensorWatch& watch :
       debug_options.debug_tensor_watch_opts()) {
    for (const string& url : watch.debug_urls()) {
      debug_urls_.insert(url);
    }
  }
  if (debug_options.reset_disk_byte_usage()) {
    DebugFileIO::resetDiskByteUsage();
  }
}

DebuggerState::~DebuggerState() {
  for (const string& debug_url : debug_urls_) {
    DebugIO::CloseDebugURL(debug_url).IgnoreError();
  }
}

absl::Status DebuggerState::PublishDebugMetadata(
    const int64_t global_step, const int64_t session_run_index,
    const int64_t executor_step_index, const std::vector<string>& input_names,
    const std::vector<string>& output_names,
    const std::vector<string>& target_names) {
  return DebugIO::PublishDebugMetadata(global_step, session_run_index,
                                       executor_step_index, input_names,
                                       output_names, target_names, debug_urls_);
}

absl::Status DebugGraphDecorator::DecorateGraph(Graph* graph, Device* device) {
  DebugNodeInserter::DeparallelizeWhileLoops(graph, device);
  return DebugNodeInserter::InsertNodes(
      debug_options_.debug_tensor_watch_opts(), graph, device);
}

absl::Status DebugGraphDecorator::PublishGraph(const Graph& graph,
                                               const string& device_name) {
  std::unordered_set<string> debug_urls;
  for (const DebugTensorWatch& watch :
       debug_options_.debug_tensor_watch_opts()) {
    for (const string& url : watch.debug_urls()) {
      debug_urls.insert(url);
    }
  }

  return DebugIO::PublishGraph(graph, device_name, debug_urls);
}

}  // namespace machina
