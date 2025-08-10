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

#include "machina_serving/servables/machina/tfrt_classification_service.h"

#include <memory>

#include "machina/core/lib/core/errors.h"
#include "machina/core/platform/tracing.h"
#include "machina_serving/apis/classifier.h"
#include "machina_serving/core/servable_handle.h"
#include "machina_serving/model_servers/server_core.h"
#include "machina_serving/servables/machina/servable.h"
#include "machina_serving/servables/machina/tfrt_classifier.h"
#include "machina_serving/servables/machina/util.h"

namespace machina {
namespace serving {

absl::Status TFRTClassificationServiceImpl::Classify(
    const Servable::RunOptions& run_options, ServerCore* core,
    const ClassificationRequest& request, ClassificationResponse* response) {
  // Verify Request Metadata and create a ServableRequest
  if (!request.has_model_spec()) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "Missing ModelSpec");
  }

  return ClassifyWithModelSpec(run_options, core, request.model_spec(), request,
                               response);
}

absl::Status TFRTClassificationServiceImpl::ClassifyWithModelSpec(
    const Servable::RunOptions& run_options, ServerCore* core,
    const ModelSpec& model_spec, const ClassificationRequest& request,
    ClassificationResponse* response) {
  TRACELITERAL("TFRTClassificationServiceImpl::ClassifyWithModelSpec");

  ServableHandle<Servable> servable;
  TF_RETURN_IF_ERROR(core->GetServableHandle(model_spec, &servable));
  return servable->Classify(run_options, request, response);
}

}  // namespace serving
}  // namespace machina
