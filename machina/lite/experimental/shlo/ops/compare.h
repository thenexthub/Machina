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

#ifndef MACHINA_LITE_EXPERIMENTAL_SHLO_OPS_COMPARE_H_
#define MACHINA_LITE_EXPERIMENTAL_SHLO_OPS_COMPARE_H_

#include "absl/status/status.h"
#include "machina/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

struct CompareOp {
  enum class ComparisonDirection { kEq, kNe, kGe, kGt, kLe, kLt };
  // `compare_type` is considered for deprecation. We won't implement it until
  // the deprecation is lifted.
  //
  // https://github.com/openxla/stablehlo/issues/584

  struct Attributes {
    ComparisonDirection comparison_direction;
  };

  Attributes attributes;
};

CompareOp Create(CompareOp::Attributes attributes);
absl::Status Prepare(CompareOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output);
absl::Status Evaluate(CompareOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output);

}  // namespace shlo_ref

#endif  // MACHINA_LITE_EXPERIMENTAL_SHLO_OPS_COMPARE_H_
