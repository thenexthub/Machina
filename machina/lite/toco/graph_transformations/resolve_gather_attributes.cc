/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
#include <cstddef>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/status.h"
#include "machina/lite/toco/graph_transformations/graph_transformations.h"
#include "machina/lite/toco/model.h"
#include "machina/lite/toco/tooling_util.h"

namespace toco {

absl::Status ResolveGatherAttributes::Run(Model* model, std::size_t op_index,
                                          bool* modified) {
  *modified = false;
  auto* gather_op = model->operators[op_index].get();
  if (gather_op->type != OperatorType::kGather) return absl::OkStatus();
  auto* op = static_cast<GatherOperator*>(gather_op);

  if (op->axis) {
    // Attributes already resolved
    return absl::OkStatus();
  }
  if (op->inputs.size() != 3) return absl::OkStatus();
  if (!IsConstantParameterArray(*model, op->inputs[2])) return absl::OkStatus();

  const auto& indices_array = model->GetArray(op->inputs[2]);
  if (!indices_array.has_shape()) return absl::OkStatus();
  const auto& axis_data = indices_array.GetBuffer<ArrayDataType::kInt32>().data;
  CHECK_EQ(axis_data.size(), 1)
      << "Multidimensional gather not supported on " << LogName(*op);
  op->axis = {axis_data[0]};

  // Drop the axis array as we no longer need it.
  DeleteArrayIfUnusedOutsideOfOp(op->inputs[2], op, model);
  op->inputs.resize(2);

  *modified = true;
  return absl::OkStatus();
}

}  // namespace toco
