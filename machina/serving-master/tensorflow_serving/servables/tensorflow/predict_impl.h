/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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

#ifndef MACHINA_SERVING_SERVABLES_MACHINA_PREDICT_IMPL_H_
#define MACHINA_SERVING_SERVABLES_MACHINA_PREDICT_IMPL_H_

#include "machina/core/framework/tensor.pb.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/protobuf/config.pb.h"
#include "machina_serving/apis/predict.pb.h"
#include "machina_serving/model_servers/server_core.h"
#include "machina_serving/servables/machina/thread_pool_factory.h"

namespace machina {
namespace serving {

// Utility methods for implementation of PredictionService::Predict.
class TensorflowPredictor {
 public:
  TensorflowPredictor() {}

  explicit TensorflowPredictor(ThreadPoolFactory* thread_pool_factory)
      : thread_pool_factory_(thread_pool_factory) {}

  Status Predict(const RunOptions& run_options, ServerCore* core,
                 const PredictRequest& request, PredictResponse* response);

  // Like Predict(), but uses 'model_spec' instead of the one embedded in
  // 'request'.
  Status PredictWithModelSpec(const RunOptions& run_options, ServerCore* core,
                              const ModelSpec& model_spec,
                              const PredictRequest& request,
                              PredictResponse* response);

 private:
  ThreadPoolFactory* thread_pool_factory_ = nullptr;
};

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_SERVABLES_MACHINA_PREDICT_IMPL_H_
