/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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

#include "machina_serving/servables/machina/classification_service.h"

#include <memory>

#include "machina/core/lib/core/errors.h"
#include "machina/core/platform/tracing.h"
#include "machina_serving/apis/classifier.h"
#include "machina_serving/core/servable_handle.h"
#include "machina_serving/servables/machina/classifier.h"
#include "machina_serving/servables/machina/util.h"

namespace machina {
namespace serving {

absl::Status TensorflowClassificationServiceImpl::Classify(
    const RunOptions& run_options, ServerCore* core,
    const thread::ThreadPoolOptions& thread_pool_options,
    const ClassificationRequest& request, ClassificationResponse* response) {
  // Verify Request Metadata and create a ServableRequest
  if (!request.has_model_spec()) {
    return absl::Status(
        static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument),
        "Missing ModelSpec");
  }

  return ClassifyWithModelSpec(run_options, core, thread_pool_options,
                               request.model_spec(), request, response);
}

absl::Status TensorflowClassificationServiceImpl::ClassifyWithModelSpec(
    const RunOptions& run_options, ServerCore* core,
    const thread::ThreadPoolOptions& thread_pool_options,
    const ModelSpec& model_spec, const ClassificationRequest& request,
    ClassificationResponse* response) {
  TRACELITERAL("TensorflowClassificationServiceImpl::ClassifyWithModelSpec");

  ServableHandle<SavedModelBundle> saved_model_bundle;
  TF_RETURN_IF_ERROR(core->GetServableHandle(model_spec, &saved_model_bundle));
  return RunClassify(run_options, saved_model_bundle->meta_graph_def,
                     saved_model_bundle.id().version,
                     saved_model_bundle->session.get(), request, response,
                     thread_pool_options);
}

}  // namespace serving
}  // namespace machina
