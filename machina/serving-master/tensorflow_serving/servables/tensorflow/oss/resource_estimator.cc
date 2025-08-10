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

#include "machina_serving/servables/machina/oss/resource_estimator.h"

#include "machina/core/platform/path.h"
#include "machina_serving/resources/resource_values.h"
#include "machina_serving/servables/machina/util.h"

namespace machina {
namespace serving {

absl::Status EstimateMainRamBytesFromValidationResult(
    const string& path, ResourceAllocation* estimate) {
  return errors::Unimplemented("Reading validation results is not supported");
}

absl::Status EstimateMainRamBytesFromPath(const string& path,
                                          bool use_validation_result,
                                          FileProbingEnv* env,
                                          ResourceAllocation* estimate) {
  return EstimateResourceFromPathUsingDiskState(path, env, estimate);
}

}  // namespace serving
}  // namespace machina
