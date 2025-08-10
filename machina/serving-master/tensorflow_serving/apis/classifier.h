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

#ifndef MACHINA_SERVING_APIS_CLASSIFIER_H_
#define MACHINA_SERVING_APIS_CLASSIFIER_H_

#include "machina/core/lib/core/status.h"
#include "machina_serving/apis/classification.pb.h"

namespace machina {
namespace serving {

/// Model-type agnostic interface for performing classification.
///
/// Specific implementations will exist for different model types
/// (e.g. TensorFlow SavedModel) that can convert the request into a model
/// specific input and know how to convert the output into a generic
/// ClassificationResult.
class ClassifierInterface {
 public:
  /// Given a ClassificationRequest, populates the ClassificationResult with the
  /// result.
  ///
  /// @param request  Input request specifying the model/signature to query
  /// along with the data payload.
  /// @param result   The output classifications that will get populated.
  /// @return         A status object indicating success or failure.
  virtual Status Classify(const ClassificationRequest& request,
                          ClassificationResult* result) = 0;

  virtual ~ClassifierInterface() = default;
};

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_APIS_CLASSIFIER_H_
