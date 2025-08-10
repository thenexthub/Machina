/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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

#include "machina/lite/experimental/shlo/ops/sign.h"

#include "absl/status/status.h"
#include "machina/lite/experimental/shlo/bf16.h"
#include "machina/lite/experimental/shlo/dispatch.h"
#include "machina/lite/experimental/shlo/f16.h"
#include "machina/lite/experimental/shlo/ops/unary_elementwise.h"
#include "machina/lite/experimental/shlo/ops/util.h"
#include "machina/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

struct Sign {
  template <class T>
  T operator()(T v) const {
    constexpr T one = static_cast<T>(1);
    constexpr T minus_one = static_cast<T>(-1);
    constexpr T zero = static_cast<T>(0);
    return v < zero ? minus_one : (v > zero ? one : v);
  }
};

template <>
F16 Sign::operator()(F16 v) const {
  return static_cast<F16>(operator()(static_cast<float>(v)));
}

template <>
BF16 Sign::operator()(BF16 v) const {
  return static_cast<BF16>(operator()(static_cast<float>(v)));
}

SignOp Create(SignOp::Attributes) { return {}; }

absl::Status Prepare(SignOp& op, const Tensor& input, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(input.shape(), output.shape()));

  SHLO_REF_RETURN_ON_ERROR(CheckSupportedTypes(CheckCtx("sign"), input,
                                               IsSignedIntTensor, IsFloatTensor,
                                               IsQuantizedPerTensorTensor));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("sign"), input, output));
  return absl::OkStatus();
}

absl::Status Evaluate(SignOp& op, const Tensor& input, Tensor& output) {
  Sign sign;
  if (input.IsPerTensorQuantized()) {
    DISPATCH_QUANTIZED(
        detail::DequantizeOpQuantizePerTensor,
        input.quantized_per_tensor_element_type().StorageType(),
        input.quantized_per_tensor_element_type().ExpressedType(), sign, input,
        output)
  } else if (IsSignedIntTensor(input) || IsFloatTensor(input)) {
    DISPATCH_INT_FLOAT(detail::EvaluateNoQuantization,
                       input.tensor_element_type(), sign, input, output);
  }
  return absl::FailedPreconditionError(
      "stablehlo.sign: Unsupported tensor type.");
}

};  // namespace shlo_ref
