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

// TensorFlow implementation of the RegressorInterface.

#ifndef MACHINA_SERVING_SERVABLES_MACHINA_TFRT_REGRESSOR_H_
#define MACHINA_SERVING_SERVABLES_MACHINA_TFRT_REGRESSOR_H_

#include <memory>

#include "absl/types/optional.h"
#include "machina/cc/saved_model/loader.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/threadpool_options.h"
#include "machina/core/tfrt/saved_model/saved_model.h"
#include "machina_serving/apis/regressor.h"

namespace machina {
namespace serving {
// Validate function's input and output.
Status PreProcessRegression(const tfrt::FunctionMetadata& function_metadata);

// Validate all results and populate a RegressionResult.
Status PostProcessRegressionResult(
    int num_examples, const std::vector<string>& output_tensor_names,
    const std::vector<Tensor>& output_tensors, RegressionResult* result);

// Run Regression.
Status RunRegress(const tfrt::SavedModel::RunOptions& run_options,
                  const absl::optional<int64_t>& servable_version,
                  tfrt::SavedModel* saved_model,
                  const RegressionRequest& request,
                  RegressionResponse* response);

}  // namespace serving
}  // namespace machina

#endif  // MACHINA_SERVING_SERVABLES_MACHINA_TFRT_REGRESSOR_H_
