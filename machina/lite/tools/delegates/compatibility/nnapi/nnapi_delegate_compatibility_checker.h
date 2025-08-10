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

#ifndef MACHINA_LITE_TOOLS_DELEGATES_COMPATIBILITY_NNAPI_NNAPI_DELEGATE_COMPATIBILITY_CHECKER_H_
#define MACHINA_LITE_TOOLS_DELEGATES_COMPATIBILITY_NNAPI_NNAPI_DELEGATE_COMPATIBILITY_CHECKER_H_

#include <string>
#include <unordered_map>

#include "absl/status/status.h"
#include "machina/lite/c/common.h"
#include "machina/lite/model_builder.h"
#include "machina/lite/tools/delegates/compatibility/common/delegate_compatibility_checker_base.h"
#include "machina/lite/tools/delegates/compatibility/protos/compatibility_result.pb.h"
#include "machina/lite/tools/versioning/op_signature.h"

namespace tflite {
namespace tools {

// Default runtime feature level used when no parameters are specified.
inline constexpr int kDefaultRuntimeFeatureLevel = 8;

// Class to check if an operation or a model is compatible with NNAPI delegate.
// Supported parameters:
//   - nnapi-runtime_feature_level: Between 1 and 8 (default value: 8)
class NnapiDelegateCompatibilityChecker
    : public DelegateCompatibilityCheckerBase {
 public:
  NnapiDelegateCompatibilityChecker() {
    runtime_feature_level_ = kDefaultRuntimeFeatureLevel;
  }

  absl::Status checkModelCompatibilityOnline(
      tflite::FlatBufferModel* model_buffer,
      tflite::proto::CompatibilityResult* result) override;

  // Checks if the node is compatible with the NNAPI delegate using online mode.
  // Params:
  //   context: Used to get the tensors. TfLiteTensors can be obtained via
  //         TfLiteContext, which are used to get tensor type and tensor data,
  //         the same way as with OpSignature in Offline mode.
  //   node: Used with context to get the desired tensor, e.g.:
  //         context->tensors[node->inputs->data[0]]
  //   registration: Used to get the builtin code and the operator version.
  //   op_result: Used to store if the node is compatible with the delegate or
  //              not and why (with a human readable message).
  static absl::Status checkOpCompatibilityOnline(
      TfLiteContext* context, const TfLiteNode* node,
      const TfLiteRegistration* registration,
      std::unordered_map<std::string, std::string> dcc_configs,
      tflite::proto::OpCompatibilityResult* op_result);

  // Returns a dictionary with NNAPI delegate specific params.
  // Keys:
  //   - nnapi-runtime_feature_level
  std::unordered_map<std::string, std::string> getDccConfigurations() override;

  // Sets the parameters needed in the specific DCC.
  // Keys:
  //   - nnapi-runtime_feature_level
  absl::Status setDccConfigurations(
      const std::unordered_map<std::string, std::string>& dcc_configs) override;

 private:
  absl::Status checkOpSigCompatibility(
      const OpSignature& op_sig,
      tflite::proto::OpCompatibilityResult* op_result) override;

  // Runtime feature level
  // Refer to '/machina/lite/nnapi/NeuralNetworksTypes.h'
  int runtime_feature_level_;
};

}  // namespace tools
}  // namespace tflite

#endif  // MACHINA_LITE_TOOLS_DELEGATES_COMPATIBILITY_NNAPI_NNAPI_DELEGATE_COMPATIBILITY_CHECKER_H_
