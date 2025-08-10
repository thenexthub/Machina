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

#ifndef MACHINA_SERVING_SERVABLES_MACHINA_CLASSIFICATION_SERVICE_H_
#define MACHINA_SERVING_SERVABLES_MACHINA_CLASSIFICATION_SERVICE_H_

#include "machina/core/lib/core/status.h"
#include "machina/core/platform/threadpool_options.h"
#include "machina/core/protobuf/config.pb.h"
#include "machina_serving/apis/classification.pb.h"
#include "machina_serving/model_servers/server_core.h"

namespace machina {
namespace serving {

// Utility methods for implementation of
// machina_serving/apis/classification-service.proto.
class TensorflowClassificationServiceImpl {
 public:
  static Status Classify(const RunOptions& run_options, ServerCore* core,
                         const thread::ThreadPoolOptions& thread_pool_options,
                         const ClassificationRequest& request,
                         ClassificationResponse* response);

  // Like Classify(), but uses 'model_spec' instead of the one embedded in
  // 'request'.
  static Status ClassifyWithModelSpec(
      const RunOptions& run_options, ServerCore* core,
      const thread::ThreadPoolOptions& thread_pool_options,
      const ModelSpec& model_spec, const ClassificationRequest& request,
      ClassificationResponse* response);
};

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_SERVABLES_MACHINA_CLASSIFICATION_SERVICE_H_
