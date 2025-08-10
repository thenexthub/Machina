/* Copyright 2017 Google Inc. All Rights Reserved.

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

#ifndef MACHINA_SERVING_SERVABLES_MACHINA_MULTI_INFERENCE_H_
#define MACHINA_SERVING_SERVABLES_MACHINA_MULTI_INFERENCE_H_

#include "absl/types/optional.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/threadpool_options.h"
#include "machina/core/protobuf/meta_graph.pb.h"
#include "machina/core/public/session.h"
#include "machina_serving/apis/inference.pb.h"

namespace machina {
namespace serving {

// TensorFlow implementation of the MultiInference.
// Only supports Models in the SavedModel format.
class TensorFlowMultiInferenceRunner {
 public:
  TensorFlowMultiInferenceRunner(Session* session,
                                 const MetaGraphDef* meta_graph_def)
      : TensorFlowMultiInferenceRunner(session, meta_graph_def,
                                       /*servable_version=*/{}) {}

  TensorFlowMultiInferenceRunner(
      Session* session, const MetaGraphDef* meta_graph_def,
      absl::optional<int64_t> servable_version,
      const thread::ThreadPoolOptions& thread_pool_options =
          thread::ThreadPoolOptions())
      : session_(session),
        meta_graph_def_(meta_graph_def),
        servable_version_(servable_version),
        thread_pool_options_(thread_pool_options) {}

  // Run inference and return the inference results in the same order as the
  // InferenceTasks in the request.
  Status Infer(const RunOptions& run_options,
               const MultiInferenceRequest& request,
               MultiInferenceResponse* response);

  virtual ~TensorFlowMultiInferenceRunner() = default;

 private:
  Session* const session_;
  const MetaGraphDef* const meta_graph_def_;
  // If available, servable_version is used to set the ModelSpec version in the
  // InferenceResults of the MultiInferenceResponse.
  const absl::optional<int64_t> servable_version_;
  const machina::thread::ThreadPoolOptions thread_pool_options_;
};

// Creates TensorFlowMultiInferenceRunner and calls Infer on it.
Status RunMultiInference(
    const RunOptions& run_options, const MetaGraphDef& meta_graph_def,
    const absl::optional<int64_t>& servable_version, Session* session,
    const MultiInferenceRequest& request, MultiInferenceResponse* response,
    const machina::thread::ThreadPoolOptions& thread_pool_options =
        machina::thread::ThreadPoolOptions());

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_SERVABLES_MACHINA_MULTI_INFERENCE_H_
