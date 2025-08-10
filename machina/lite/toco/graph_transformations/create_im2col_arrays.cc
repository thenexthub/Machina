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
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/status.h"
#include "machina/lite/toco/graph_transformations/graph_transformations.h"
#include "machina/lite/toco/model.h"
#include "machina/lite/toco/tooling_util.h"

namespace toco {

bool ProcessConvOperator(Model* model, ConvOperator* op) {
  if (op->outputs.size() == 2) {
    // We already have an im2col array
    return false;
  }
  const auto& weights_array = model->GetArray(op->inputs[1]);
  if (!weights_array.has_shape()) {
    // We need to yield until weights dims have been resolved, because
    // from the weights dims we determine whether an im2col array is
    // needed.
    return false;
  }
  const auto& weights_shape = weights_array.shape();
  const int kheight = weights_shape.dims(1);
  const int kwidth = weights_shape.dims(2);
  if (kwidth == 1 && kheight == 1 && op->stride_width == 1 &&
      op->stride_height == 1 && op->dilation_width_factor == 1 &&
      op->dilation_height_factor == 1) {
    // 1x1 unstrided undilated conv does not need an im2col array.
    return false;
  }

  // Create the im2col array.
  CHECK_EQ(op->outputs.size(), 1);
  const std::string& im2col_array_name =
      AvailableArrayName(*model, op->inputs[0] + "_im2col");
  model->GetOrCreateArray(im2col_array_name);
  op->outputs.push_back(im2col_array_name);

  return true;
}

bool ProcessTransposeConvOperator(Model* model, TransposeConvOperator* op) {
  if (op->outputs.size() == 2) {
    // We already have an im2col array
    return false;
  }

  // Always create an im2col array for transpose_conv.
  CHECK_EQ(op->outputs.size(), 1);
  const std::string& im2col_array_name = AvailableArrayName(
      *model, op->inputs[TransposeConvOperator::DATA_INPUT] + "_im2col");
  model->GetOrCreateArray(im2col_array_name);
  op->outputs.push_back(im2col_array_name);

  return true;
}

absl::Status CreateIm2colArrays::Run(Model* model, std::size_t op_index,
                                     bool* modified) {
  *modified = false;
  auto it = model->operators.begin() + op_index;
  auto* op = it->get();

  switch (op->type) {
    case OperatorType::kConv:
      *modified = ProcessConvOperator(model, static_cast<ConvOperator*>(op));
      return absl::OkStatus();
    case OperatorType::kTransposeConv:
      *modified = ProcessTransposeConvOperator(
          model, static_cast<TransposeConvOperator*>(op));
      return absl::OkStatus();
    default:
      return absl::OkStatus();
  }
}

}  // namespace toco
