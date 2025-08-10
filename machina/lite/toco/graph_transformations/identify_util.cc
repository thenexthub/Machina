/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include "machina/lite/toco/graph_transformations/identify_util.h"

#include <string>

#include "machina/lite/toco/model.h"
#include "machina/lite/toco/runtime/types.h"
#include "machina/lite/toco/tooling_util.h"

namespace toco {
namespace util {

bool IsBinaryOp(const Operator* op, OperatorType optype,
                FusedActivationFunctionType act) {
  return op && op->type == optype && op->inputs.size() == 2 &&
         op->fused_activation_function == act;
}

bool CheckArrayIsScalarFloat(Model* model, const std::string& name, float val) {
  const auto& op_array = model->GetArray(name);
  if (!op_array.buffer || op_array.buffer->type != ArrayDataType::kFloat ||
      RequiredBufferSizeForShape(op_array.shape()) != 1) {
    return false;
  }
  const auto& op_data = op_array.GetBuffer<ArrayDataType::kFloat>().data;
  return op_data[0] == val;
}

int GetSingleScalarInputIndexOfBinaryOp(Model* model, const Operator* op,
                                        float val) {
  bool input0_is_scalar = CheckArrayIsScalarFloat(model, op->inputs[0], val);
  bool input1_is_scalar = CheckArrayIsScalarFloat(model, op->inputs[1], val);
  return input0_is_scalar == input1_is_scalar ? -1 : input0_is_scalar ? 0 : 1;
}

}  // namespace util
}  // namespace toco
