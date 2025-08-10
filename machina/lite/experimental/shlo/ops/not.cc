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

#include "machina/lite/experimental/shlo/ops/not.h"

#include "absl/status/status.h"
#include "machina/lite/experimental/shlo/dispatch.h"
#include "machina/lite/experimental/shlo/ops/unary_elementwise.h"
#include "machina/lite/experimental/shlo/ops/util.h"
#include "machina/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

struct Not {
  template <class T>
  T operator()(T v) const {
    return static_cast<T>(~v);
  }
};

template <>
bool Not::operator()(bool v) const {
  return !v;
}

NotOp Create(NotOp::Attributes) { return {}; }

absl::Status Prepare(NotOp& op, const Tensor& input, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(input.shape(), output.shape()));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSupportedTypes(CheckCtx("not"), input, IsBoolTensor, IsIntTensor));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("not"), input, output));
  return absl::OkStatus();
}

absl::Status Evaluate(NotOp& op, const Tensor& input, Tensor& output) {
  Not not_func;
  if (IsIntTensor(input) || IsBoolTensor(input)) {
    DISPATCH_BOOL_INT(detail::EvaluateNoQuantization,
                      input.tensor_element_type(), not_func, input, output);
  }
  return absl::FailedPreconditionError(
      "stablehlo.not: Unsupported tensor type.");
}

};  // namespace shlo_ref
