/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/status.h"
#include "machina/lite/toco/graph_transformations/graph_transformations.h"
#include "machina/lite/toco/model.h"
#include "machina/lite/toco/tooling_util.h"

namespace toco {

namespace {

int GetOutputDepthFromWeights(const Model& model, const Operator& op) {
  const std::string& weights_name = op.inputs[1];
  const auto& weights_shape = model.GetArray(weights_name).shape();
  if (op.type == OperatorType::kConv ||
      op.type == OperatorType::kFullyConnected ||
      op.type == OperatorType::kTransposeConv) {
    return weights_shape.dims(0);
  }
  if (op.type == OperatorType::kDepthwiseConv) {
    return weights_shape.dims(3);
  }
  LOG(FATAL) << "Unhandled operator type";
  return 0;
}

bool CheckOpInputSize(const Operator& op) {
  if (op.type == OperatorType::kConv ||
      op.type == OperatorType::kFullyConnected ||
      op.type == OperatorType::kDepthwiseConv) {
    return (op.inputs.size() >= 3);
  } else if (op.type == OperatorType::kTransposeConv) {
    return (op.inputs.size() >= 4);
  }
  return true;
}

bool ProcessLinearOperator(Model* model, Operator* op) {
  if (CheckOpInputSize(*op)) {
    return false;
  }
  const std::string& output_name = op->outputs[0];
  const std::string& weights_name = op->inputs[1];
  if (!model->GetArray(weights_name).has_shape()) {
    return false;
  }
  const int depth = GetOutputDepthFromWeights(*model, *op);
  const std::string& bias_name =
      AvailableArrayName(*model, output_name + "_bias");
  op->inputs.push_back(bias_name);
  auto& bias_array = model->GetOrCreateArray(bias_name);
  bias_array.data_type = ArrayDataType::kFloat;
  bias_array.mutable_shape()->mutable_dims()->push_back(depth);
  auto& bias_buffer = bias_array.GetMutableBuffer<ArrayDataType::kFloat>();
  bias_buffer.data.resize(depth, 0.f);
  return true;
}
}  // namespace

absl::Status EnsureBiasVectors::Run(Model* model, std::size_t op_index,
                                    bool* modified) {
  *modified = false;
  auto* op = model->operators[op_index].get();
  if (op->type == OperatorType::kConv ||
      op->type == OperatorType::kDepthwiseConv ||
      op->type == OperatorType::kFullyConnected ||
      op->type == OperatorType::kTransposeConv) {
    if (ProcessLinearOperator(model, op)) {
      AddMessageF("Added bias vector to %s as %s", LogName(*op), op->inputs[2]);
      *modified = true;
      return absl::OkStatus();
    }
  }
  return absl::OkStatus();
}

}  // namespace toco
