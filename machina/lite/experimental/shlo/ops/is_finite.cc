/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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
#include "machina/lite/experimental/shlo/ops/is_finite.h"

#include <algorithm>
#include <cmath>

#include "absl/status/status.h"
#include "machina/lite/experimental/shlo/data_type.h"
#include "machina/lite/experimental/shlo/dispatch.h"
#include "machina/lite/experimental/shlo/shape.h"
#include "machina/lite/experimental/shlo/tensor.h"

namespace shlo_ref {
namespace {

absl::Status CheckParameters(const Tensor& operand, const Tensor& result) {
  if (operand.shape() != result.shape()) {
    return absl::InvalidArgumentError(
        "operand and result must have the same shape");
  }
  if (!operand.IsQuantized() && !IsFloat(operand.tensor_element_type())) {
    return absl::InvalidArgumentError(
        "operand must be floating-point type or per-tensor quantized");
  }
  if (operand.IsPerAxisQuantized()) {
    return absl::InvalidArgumentError("operand cannot be per-axis quantized");
  }
  if (result.IsQuantized()) {
    return absl::InvalidArgumentError("result cannot be quantized");
  }
  if (!IsBool(result.tensor_element_type())) {
    return absl::InvalidArgumentError("result must be an I1 tensor");
  }
  if (operand.NumElements() != result.NumElements()) {
    return absl::InvalidArgumentError(
        "operand and result must have the same size");
  }
  return absl::OkStatus();
}

template <DataType data_type>
absl::Status EvaluateImpl(const Tensor& operand, bool* output) {
  const auto* in = operand.GetDataAs<data_type>();
  const auto num_elements = operand.NumElements();
  for (DimensionSize i = 0; i < num_elements; ++i) {
    output[i] = std::isfinite(static_cast<float>(in[i]));
  }
  return absl::OkStatus();
}

absl::Status EvaluateImpl(const Tensor& operand, Tensor& result) {
  bool* output = result.GetDataAs<DataType::kI1>();
  if (!operand.IsQuantized()) {
    DISPATCH_FLOAT(EvaluateImpl, operand.tensor_element_type(), operand,
                   output);
  } else {  // IsQuantized(operand)
    // For quantized types, the result is always true.
    const auto num_elements = result.NumElements();
    std::fill(output, output + num_elements, true);
  }
  return absl::OkStatus();
}

}  // namespace

IsFiniteOp Create(const IsFiniteOp::Attributes& attributes) {
  return IsFiniteOp();
}

absl::Status Prepare(IsFiniteOp& op, const Tensor& operand, Tensor& result) {
  return CheckParameters(operand, result);
}

absl::Status Evaluate(IsFiniteOp& op, const Tensor& operand, Tensor& result) {
  if (!operand.data) {
    return absl::InvalidArgumentError("No operand.data");
  }
  if (!result.data) {
    return absl::InvalidArgumentError("No result.data");
  }
  return EvaluateImpl(operand, result);
}

}  // namespace shlo_ref
