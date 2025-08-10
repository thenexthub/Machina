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
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/status.h"
#include "machina/lite/toco/graph_transformations/graph_transformations.h"
#include "machina/lite/toco/graph_transformations/quantization_util.h"
#include "machina/lite/toco/graph_transformations/remove_trivial_passthrough.h"
#include "machina/lite/toco/model.h"
#include "machina/lite/toco/runtime/types.h"
#include "machina/lite/toco/tooling_util.h"

namespace toco {

namespace {

bool IsTrivialUnfusedActivationFunc(GraphTransformation* transformation,
                                    const Model& model, OperatorType op_type,
                                    const std::string& input_array_name) {
  double clamp_min;
  double clamp_max;
  switch (op_type) {
    case OperatorType::kRelu:
      clamp_min = 0.0;
      clamp_max = std::numeric_limits<double>::infinity();
      break;
    case OperatorType::kRelu1:
      clamp_min = -1.0;
      clamp_max = 1.0;
      break;
    case OperatorType::kRelu6:
      clamp_min = 0.0;
      clamp_max = 6.0;
      break;
    default:
      return false;
  }

  const auto& input_array = model.GetArray(input_array_name);
  return IsArrayQuantizedRangeSubset(transformation, input_array, clamp_min,
                                     clamp_max);
}

bool IsTrivialFusedActivationFunc(
    GraphTransformation* transformation, const Model& model,
    FusedActivationFunctionType activation_function,
    const std::string& output_array_name) {
  double clamp_min;
  double clamp_max;
  switch (activation_function) {
    case FusedActivationFunctionType::kNone:
      return false;
    case FusedActivationFunctionType::kRelu:
      clamp_min = 0.0;
      clamp_max = std::numeric_limits<double>::infinity();
      break;
    case FusedActivationFunctionType::kRelu1:
      clamp_min = -1.0;
      clamp_max = 1.0;
      break;
    case FusedActivationFunctionType::kRelu6:
      clamp_min = 0.0;
      clamp_max = 6.0;
      break;
    default:
      LOG(FATAL) << "Unsupported fused activation type: "
                 << static_cast<int>(activation_function);
      return false;
  }

  const auto& output_array = model.GetArray(output_array_name);
  return IsArrayQuantizedRangeSubset(transformation, output_array, clamp_min,
                                     clamp_max);
}

}  // namespace

// Attempts to remove both fused and unfused activation functions if the
// quantization params indicate that the representable values fall inside the
// activation range.
absl::Status RemoveTrivialQuantizedActivationFunc::Run(Model* model,
                                                       std::size_t op_index,
                                                       bool* modified) {
  *modified = false;
  const auto it = model->operators.begin() + op_index;
  auto* op = it->get();
  if (op->inputs.empty()) {
    return absl::OkStatus();
  }

  if (IsTrivialUnfusedActivationFunc(this, *model, op->type, op->inputs[0])) {
    AddMessageF(
        "Removing trivial unfused activation function %s because the input "
        "minmax imply at least as tight a clamp anyway.",
        LogName(*op));
    *modified = RemoveTrivialPassthroughOp(this, model, op_index);
    return absl::OkStatus();
  }
  if (IsTrivialFusedActivationFunc(this, *model, op->fused_activation_function,
                                   op->outputs[0])) {
    op->fused_activation_function = FusedActivationFunctionType::kNone;
    AddMessageF(
        "Removing trivial quantized activation function on %s "
        "because the output quantization parameters imply at least as tight "
        "a clamp anyway.",
        LogName(*op));
    *modified = true;
    return absl::OkStatus();
  }
  return absl::OkStatus();
}

}  // namespace toco
