/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#ifndef MACHINA_TOOLS_BENCHMARK_BENCHMARK_MODEL_H_
#define MACHINA_TOOLS_BENCHMARK_BENCHMARK_MODEL_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/types.h"
#include "machina/core/public/session.h"
#include "machina/core/util/stat_summarizer.h"

namespace machina {
namespace benchmark_model {

// Used to help construct dummy inputs for the benchmarking.
struct InputLayerInfo {
  string name;
  DataType data_type;
  TensorShape shape;
  std::vector<float> initialization_values;
};

// Loads a model from disk into a new session.
absl::Status InitializeSession(int num_threads, const string& graph,
                               std::unique_ptr<Session>* session,
                               std::unique_ptr<GraphDef>* graph_def);

// Does a single run of the model that's been loaded into the given session.
absl::Status RunBenchmark(const std::vector<InputLayerInfo>& inputs,
                          const std::vector<string>& outputs,
                          const std::vector<string>& targets, Session* session,
                          StatSummarizer* stats, int64_t* inference_time_us);

// Runs the model multiple time, keeping track of timing information.
absl::Status TimeMultipleRuns(double sleep_seconds, int num_runs,
                              double max_time_s,
                              const std::vector<InputLayerInfo>& inputs,
                              const std::vector<string>& outputs,
                              const std::vector<string>& targets,
                              Session* session, StatSummarizer* stats,
                              int64_t* total_time_us, int64_t* actual_num_runs);

// Handles all setup and argument parsing.
int Main(int argc, char** argv);

}  // namespace benchmark_model
}  // namespace machina

#endif  // MACHINA_TOOLS_BENCHMARK_BENCHMARK_MODEL_H_
