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
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "machina/core/platform/status.h"
#include "machina/lite/toco/graph_transformations/graph_transformations.h"
#include "machina/lite/toco/model.h"
#include "machina/lite/toco/model_flags.pb.h"
#include "machina/lite/toco/tooling_util.h"

namespace toco {

absl::Status RemoveFinalDequantizeOp::Run(Model* model, std::size_t op_index,
                                          bool* modified) {
  *modified = false;
  const auto dequantize_it = model->operators.begin() + op_index;
  const auto* dequantize_op = dequantize_it->get();
  if (dequantize_op->type != OperatorType::kDequantize) {
    return absl::OkStatus();
  }
  const auto& output = dequantize_op->outputs[0];
  // We can remove any dequantize op whose output is not consumed by
  // any op. This is not necessarily equivalent to the output being
  // one of the model's output arrays, as some intermediate array
  // in the middle of the graph might be designated as an output
  // array.
  if (CountOpsWithInput(*model, output)) {
    return absl::OkStatus();
  }

  // If one of the model's output arrays was actually the Dequantize op's
  // output, then we need to update it to point to the Dequantize op's input.
  for (int i = 0; i < model->flags.output_arrays_size(); i++) {
    if (output == model->flags.output_arrays(i)) {
      model->flags.set_output_arrays(i, dequantize_op->inputs[0]);
    }
  }

  // Remove the node and its output array.
  AddMessageF("Removed final %s", LogName(*dequantize_op));
  DeleteOpAndArrays(model, dequantize_op);
  *modified = true;
  return absl::OkStatus();
}

}  // namespace toco
