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

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/status.h"
#include "machina/lite/toco/graph_transformations/graph_transformations.h"
#include "machina/lite/toco/model.h"
#include "machina/lite/toco/tooling_util.h"

namespace toco {

absl::Status DropIm2colArrays::Run(Model* model, std::size_t op_index,
                                   bool* modified) {
  *modified = false;
  auto conv_it = model->operators.begin() + op_index;
  if (conv_it->get()->type != OperatorType::kConv) {
    return absl::OkStatus();
  }
  auto* conv_op = static_cast<ConvOperator*>(conv_it->get());
  if (conv_op->outputs.size() < 2) {
    // Conv op does not have im2col.
    return absl::OkStatus();
  }

  // Drop the im2col array.
  CHECK_EQ(conv_op->outputs.size(), 2);
  model->EraseArray(conv_op->outputs[1]);
  conv_op->outputs.resize(1);
  AddMessageF("Dropped an im2col array for %s", LogName(*conv_op));

  *modified = true;
  return absl::OkStatus();
}

}  // namespace toco
