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

#include "machina/lite/tools/delegates/compatibility/common/online_helper_delegate.h"

#include "machina/lite/c/c_api_types.h"
#include "machina/lite/c/common.h"
#include "machina/lite/context_util.h"
#include "machina/lite/logger.h"
#include "machina/lite/minimal_logging.h"
#include "machina/lite/tools/delegates/compatibility/protos/compatibility_result.pb.h"

namespace tflite {
namespace tools {

TfLiteStatus OnlineHelperDelegate::DoPrepare(TfLiteContext* context,
                                             TfLiteDelegate* delegate) {
  auto self = reinterpret_cast<OnlineHelperDelegate*>(delegate);
  // Gets execution plan.
  TfLiteIntArray* execution_plan;
  TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &execution_plan));

  // Validates compatibility for each node.
  for (auto node_index : TfLiteIntArrayView(execution_plan)) {
    proto::OpCompatibilityResult* op_result =
        self->result_->add_compatibility_results();
    TfLiteNode* node;
    TfLiteRegistration* registration;
    TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
        context, node_index, &node, &registration));
    // Always subgraph #0 because in the interpreter (and TfLiteContext), the
    // methods refer to the primary_subgraph, so the only subgraph available is
    // the subgraph #0.
    op_result->set_subgraph_index_in_model(0);
    op_result->set_operator_index_in_subgraph(node_index);

    auto status = self->check_op_func_ptr_(context, node, registration,
                                           self->dcc_configs_, op_result);
    if (!status.ok()) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Error at node %d: %s", node_index,
                      status.message());
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace tools
}  // namespace tflite
