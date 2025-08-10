/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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
#include <string>

#include "json/json.h"
#include "machina/lite/schema/schema_utils.h"
#include "machina/lite/tools/list_flex_ops.h"

namespace tflite {
namespace flex {

std::string OpListToJSONString(const OpKernelSet& flex_ops) {
  Json::Value result(Json::arrayValue);
  for (const OpKernel& op : flex_ops) {
    result.append(Json::Value(op.op_name));
  }
  return Json::FastWriter().write(result);
}

void AddFlexOpsFromModel(const tflite::Model* model, OpKernelSet* flex_ops) {
  auto* subgraphs = model->subgraphs();
  if (!subgraphs) return;

  for (int subgraph_index = 0; subgraph_index < subgraphs->size();
       ++subgraph_index) {
    const tflite::SubGraph* subgraph = subgraphs->Get(subgraph_index);
    auto* operators = subgraph->operators();
    auto* opcodes = model->operator_codes();
    if (!operators || !opcodes) continue;

    for (int i = 0; i < operators->size(); ++i) {
      const tflite::Operator* op = operators->Get(i);
      const tflite::OperatorCode* opcode = opcodes->Get(op->opcode_index());
      if (tflite::GetBuiltinCode(opcode) != tflite::BuiltinOperator_CUSTOM ||
          !tflite::IsFlexOp(opcode->custom_code()->c_str())) {
        continue;
      }

      // Remove the "Flex" prefix from op name.
      std::string flex_op_name(opcode->custom_code()->c_str());
      std::string tf_op_name =
          flex_op_name.substr(strlen(tflite::kFlexCustomCodePrefix));

      flex_ops->insert({tf_op_name, ""});
    }
  }
}
}  // namespace flex
}  // namespace tflite
