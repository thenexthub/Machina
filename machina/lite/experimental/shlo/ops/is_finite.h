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

#ifndef MACHINA_LITE_EXPERIMENTAL_SHLO_OPS_IS_FINITE_H_
#define MACHINA_LITE_EXPERIMENTAL_SHLO_OPS_IS_FINITE_H_

#include "absl/status/status.h"
#include "machina/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

struct IsFiniteOp {
  struct Attributes {};
};

IsFiniteOp Create(const IsFiniteOp::Attributes& attributes);
absl::Status Prepare(IsFiniteOp& op, const Tensor& operand, Tensor& result);
absl::Status Evaluate(IsFiniteOp& op, const Tensor& operand, Tensor& result);

}  // namespace shlo_ref

#endif  // MACHINA_LITE_EXPERIMENTAL_SHLO_OPS_IS_FINITE_H_
