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

#ifndef MACHINA_SERVING_SERVABLES_MACHINA_TFRT_CLASSIFICATION_SERVICE_H_
#define MACHINA_SERVING_SERVABLES_MACHINA_TFRT_CLASSIFICATION_SERVICE_H_

#include "machina/core/lib/core/status.h"
#include "machina/core/protobuf/config.pb.h"
#include "machina_serving/apis/classification.pb.h"
#include "machina_serving/model_servers/server_core.h"
#include "machina_serving/servables/machina/servable.h"

namespace machina {
namespace serving {

// Utility methods for implementation of the Classify RPC in
// machina_serving/apis/prediction_service.proto
class TFRTClassificationServiceImpl {
 public:
  static Status Classify(const Servable::RunOptions& run_options,
                         ServerCore* core, const ClassificationRequest& request,
                         ClassificationResponse* response);

  // Like Classify(), but uses 'model_spec' instead of the one embedded in
  // 'request'.
  static Status ClassifyWithModelSpec(const Servable::RunOptions& run_options,
                                      ServerCore* core,
                                      const ModelSpec& model_spec,
                                      const ClassificationRequest& request,
                                      ClassificationResponse* response);
};

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_SERVABLES_MACHINA_TFRT_CLASSIFICATION_SERVICE_H_
