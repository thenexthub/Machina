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

namespace toco {

absl::Status RemoveTensorFlowIdentity::Run(Model* model, std::size_t op_index,
                                           bool* modified) {
  *modified = false;
  const auto passthru_it = model->operators.begin() + op_index;
  const auto* passthru_op = passthru_it->get();
  if (passthru_op->type != OperatorType::kIdentity) {
    return absl::OkStatus();
  }

  *modified = RemoveTrivialPassthroughOp(this, model, op_index);
  return absl::OkStatus();
}

}  // namespace toco
