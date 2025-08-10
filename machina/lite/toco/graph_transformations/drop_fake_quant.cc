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
#include "machina/lite/toco/graph_transformations/remove_trivial_passthrough.h"
#include "machina/lite/toco/model.h"
#include "machina/lite/toco/tooling_util.h"

namespace toco {

absl::Status DropFakeQuant::Run(Model* model, std::size_t op_index,
                                bool* modified) {
  *modified = false;
  const auto fakequant_it = model->operators.begin() + op_index;
  auto* fakequant_base_op = fakequant_it->get();
  if (fakequant_base_op->type != OperatorType::kFakeQuant) {
    return absl::OkStatus();
  }
  auto* fakequant_op = static_cast<FakeQuantOperator*>(fakequant_base_op);

  if (!fakequant_op->minmax) {
    return absl::OkStatus();
  }

  const auto& output_array = model->GetArray(fakequant_op->outputs[0]);
  if (!output_array.minmax) {
    return absl::OkStatus();
  }

  // Drop min/max inputs
  for (int i = 1, end = fakequant_op->inputs.size(); i < end; i++) {
    if (CountOpsWithInput(*model, fakequant_op->inputs[i]) == 1) {
      model->EraseArray(fakequant_op->inputs[i]);
    }
  }
  fakequant_op->inputs.resize(1);

  *modified = RemoveTrivialPassthroughOp(this, model, op_index);
  return absl::OkStatus();
}

}  // namespace toco
