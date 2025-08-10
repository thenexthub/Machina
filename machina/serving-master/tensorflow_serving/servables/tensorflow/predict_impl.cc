/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "machina_serving/servables/machina/predict_impl.h"

#include <string>
#include <utility>

#include "absl/strings/substitute.h"
#include "machina/cc/saved_model/loader.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/platform/threadpool_options.h"
#include "machina_serving/core/servable_handle.h"
#include "machina_serving/servables/machina/predict_util.h"
#include "machina_serving/servables/machina/thread_pool_factory.h"
#include "machina_serving/servables/machina/util.h"

namespace machina {
namespace serving {

absl::Status TensorflowPredictor::Predict(const RunOptions& run_options,
                                          ServerCore* core,
                                          const PredictRequest& request,
                                          PredictResponse* response) {
  if (!request.has_model_spec()) {
    return absl::Status(
        static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument),
        "Missing ModelSpec");
  }
  return PredictWithModelSpec(run_options, core, request.model_spec(), request,
                              response);
}

absl::Status TensorflowPredictor::PredictWithModelSpec(
    const RunOptions& run_options, ServerCore* core,
    const ModelSpec& model_spec, const PredictRequest& request,
    PredictResponse* response) {
  ServableHandle<SavedModelBundle> bundle;
  TF_RETURN_IF_ERROR(core->GetServableHandle(model_spec, &bundle));
  return internal::RunPredict(
      run_options, bundle->meta_graph_def, bundle.id().version,
      core->predict_response_tensor_serialization_option(),
      bundle->session.get(), request, response,
      thread_pool_factory_ == nullptr
          ? thread::ThreadPoolOptions()
          : thread_pool_factory_->GetThreadPools().get());
}

}  // namespace serving
}  // namespace machina
