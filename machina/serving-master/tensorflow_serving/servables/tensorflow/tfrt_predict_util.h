/* Copyright 2020 Google Inc. All Rights Reserved.

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

#ifndef MACHINA_SERVING_SERVABLES_MACHINA_TFRT_PREDICT_UTIL_H_
#define MACHINA_SERVING_SERVABLES_MACHINA_TFRT_PREDICT_UTIL_H_

#include "absl/types/optional.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/threadpool_options.h"
#include "machina/core/protobuf/config.pb.h"
#include "machina/core/tfrt/saved_model/saved_model.h"
#include "machina_serving/apis/predict.pb.h"
#include "machina_serving/servables/machina/predict_response_tensor_serialization_option.h"

namespace machina {
namespace serving {
namespace internal {

// Similar to RunPredict below, but allows specification of a serialization
// option for the TensorProtos in the response.
Status RunPredict(
    const tfrt::SavedModel::RunOptions& run_options,
    const absl::optional<int64_t>& servable_version,
    const PredictResponseTensorSerializationOption tensor_serialization_option,
    tfrt::SavedModel* saved_model, const PredictRequest& request,
    PredictResponse* response,
    const thread::ThreadPoolOptions& thread_pool_options =
        thread::ThreadPoolOptions());

}  // namespace internal

// Implementation of Predict using the tfrt::SavedModel.
//
// IMPLEMENTATION NOTES: Calls the internal::RunPredict function above by
// specifying serialization option as kAsProtoField for backward compatibility.
Status RunPredict(const tfrt::SavedModel::RunOptions& run_options,
                  const absl::optional<int64_t>& servable_version,
                  tfrt::SavedModel* saved_model, const PredictRequest& request,
                  PredictResponse* response,
                  const thread::ThreadPoolOptions& thread_pool_options =
                      thread::ThreadPoolOptions());
}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_SERVABLES_MACHINA_TFRT_PREDICT_UTIL_H_
