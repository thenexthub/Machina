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
#include "machina/lite/experimental/shlo/ops/util.h"

#include <string>
#include <variant>

#include "absl/status/status.h"
#include "machina/lite/experimental/shlo/data_type.h"
#include "machina/lite/experimental/shlo/shape.h"
#include "machina/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

absl::Status Propagate(const Shape& input_shape, Shape& output_shape) {
  if (output_shape.Dimensions().empty()) {
    output_shape = input_shape;
  } else if (output_shape != input_shape) {
    return absl::FailedPreconditionError(
        "The specified output tensor shape is not compatible with the input "
        "shape.");
  }
  return absl::OkStatus();
}

absl::Status Propagate(const Shape& lhs_shape, const Shape& rhs_shape,
                       Shape& output_shape) {
  if (lhs_shape != rhs_shape) {
    return absl::FailedPreconditionError(
        "The LHS and RHS shapes are incompatible.");
  } else if (output_shape.Dimensions().empty()) {
    output_shape = lhs_shape;
  } else if (output_shape != lhs_shape) {
    return absl::FailedPreconditionError(
        "The specified output tensor shape is not compatible with the input "
        "shapes.");
  }
  return absl::OkStatus();
}

bool IsBoolTensor(const Tensor& tensor) {
  return !tensor.IsQuantized() && IsBool(tensor.StorageType());
}

bool IsSignedIntTensor(const Tensor& tensor) {
  return !tensor.IsQuantized() && IsSignedInteger(tensor.StorageType());
}

bool IsUnsignedIntTensor(const Tensor& tensor) {
  return !tensor.IsQuantized() && IsUnsignedInteger(tensor.StorageType());
}

bool IsIntTensor(const Tensor& tensor) {
  return !tensor.IsQuantized() && IsInteger(tensor.StorageType());
}

bool IsFloatTensor(const Tensor& tensor) {
  return !tensor.IsQuantized() && IsFloat(tensor.StorageType());
}

bool IsQuantizedPerTensorTensor(const Tensor& tensor) {
  return tensor.IsPerTensorQuantized();
}

bool IsQuantizedPerAxisTensor(const Tensor& tensor) {
  return tensor.IsPerAxisQuantized();
}

absl::Status CheckSameBaselineType(CheckCtx ctx, const Tensor& tensor1,
                                   const Tensor& tensor2) {
  if (BaselineType(tensor1.element_type()) !=
      BaselineType(tensor2.element_type())) {
    std::string tensor1_type_repr =
        std::visit([](auto v) -> std::string { return ToString(v); },
                   tensor1.element_type());
    std::string tensor2_type_repr =
        std::visit([](auto v) -> std::string { return ToString(v); },
                   tensor2.element_type());
    return absl::FailedPreconditionError(
        "stablehlo." + ctx.op_name +
        ": baseline type constraint is not satisfied " + tensor1_type_repr +
        " and " + tensor2_type_repr + ".");
  }
  return absl::OkStatus();
}

}  // namespace shlo_ref
