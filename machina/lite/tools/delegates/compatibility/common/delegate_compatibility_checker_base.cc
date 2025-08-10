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

#include "machina/lite/tools/delegates/compatibility/common/delegate_compatibility_checker_base.h"

#include <cstdlib>

#include "absl/status/status.h"
#include "machina/lite/model_builder.h"
#include "machina/lite/schema/schema_generated.h"
#include "machina/lite/tools/delegates/compatibility/common/delegate_compatibility_checker_util.h"
#include "machina/lite/tools/delegates/compatibility/protos/compatibility_result.pb.h"
#include "machina/lite/tools/versioning/op_signature.h"

namespace tflite {
namespace tools {

absl::Status DelegateCompatibilityCheckerBase::checkModelCompatibilityOffline(
    tflite::FlatBufferModel* model_buffer, proto::CompatibilityResult* result) {
  auto model = model_buffer->GetModel();
  auto subgraphs = model->subgraphs();
  for (int i = 0; i < subgraphs->Length(); ++i) {
    const tflite::SubGraph* subgraph = subgraphs->Get(i);
    for (int j = 0; j < subgraph->operators()->Length(); ++j) {
      proto::OpCompatibilityResult* op_result =
          result->add_compatibility_results();
      op_result->set_subgraph_index_in_model(i);
      op_result->set_operator_index_in_subgraph(j);
      const tflite::Operator* op = subgraph->operators()->Get(j);
      const tflite::OperatorCode* op_code =
          model->operator_codes()->Get(op->opcode_index());
      RETURN_IF_ERROR(
          checkOpCompatibilityOffline(op_code, op, subgraph, model, op_result));
    }
  }
  return absl::OkStatus();
}

absl::Status DelegateCompatibilityCheckerBase::checkOpCompatibilityOffline(
    const tflite::OperatorCode* op_code, const tflite::Operator* op,
    const tflite::SubGraph* subgraph, const tflite::Model* model,
    proto::OpCompatibilityResult* op_result) {
  OpSignature op_sig = tflite::GetOpSignature(op_code, op, subgraph, model);
  auto status = checkOpSigCompatibility(op_sig, op_result);
  if (op_sig.builtin_data) {
    free(op_sig.builtin_data);
  }
  return status;
}

}  // namespace tools
}  // namespace tflite
