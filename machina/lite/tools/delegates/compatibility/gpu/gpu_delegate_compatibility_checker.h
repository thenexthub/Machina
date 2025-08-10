/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#ifndef MACHINA_LITE_TOOLS_DELEGATES_COMPATIBILITY_GPU_GPU_DELEGATE_COMPATIBILITY_CHECKER_H_
#define MACHINA_LITE_TOOLS_DELEGATES_COMPATIBILITY_GPU_GPU_DELEGATE_COMPATIBILITY_CHECKER_H_

#include <string>
#include <unordered_map>

#include "absl/status/status.h"
#include "machina/lite/model_builder.h"
#include "machina/lite/tools/delegates/compatibility/common/delegate_compatibility_checker_base.h"
#include "machina/lite/tools/delegates/compatibility/protos/compatibility_result.pb.h"
#include "machina/lite/tools/versioning/op_signature.h"

namespace tflite {
namespace tools {

// Class to check if an operation or a model is compatible with GPU delegate.
// No supported parameters.
class GpuDelegateCompatibilityChecker
    : public DelegateCompatibilityCheckerBase {
 public:
  GpuDelegateCompatibilityChecker() {}

  // Online mode is not supported in the GPU delegate compatibility checker.
  absl::Status checkModelCompatibilityOnline(
      tflite::FlatBufferModel* model_buffer,
      tflite::proto::CompatibilityResult* result) override;

  // No parameters are supported, no need to call to this function.
  std::unordered_map<std::string, std::string> getDccConfigurations() override;

  // No parameters are supported, no need to call to this function.
  absl::Status setDccConfigurations(
      const std::unordered_map<std::string, std::string>& dcc_configs) override;

 private:
  absl::Status checkOpSigCompatibility(
      const OpSignature& op_sig,
      tflite::proto::OpCompatibilityResult* op_result) override;
};

}  // namespace tools
}  // namespace tflite

#endif  // MACHINA_LITE_TOOLS_DELEGATES_COMPATIBILITY_GPU_GPU_DELEGATE_COMPATIBILITY_CHECKER_H_
