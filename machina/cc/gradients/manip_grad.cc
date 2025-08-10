/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#include <vector>

#include "absl/status/status.h"
#include "machina/cc/framework/grad_op_registry.h"
#include "machina/cc/framework/gradients.h"
#include "machina/cc/ops/manip_ops.h"
#include "machina/cc/ops/standard_ops.h"

namespace machina {
namespace ops {
namespace {

absl::Status RollGrad(const Scope& scope, const Operation& op,
                      const std::vector<Output>& grad_inputs,
                      std::vector<Output>* grad_outputs) {
  auto shift = op.input(1);
  auto axis = op.input(2);
  auto grad_op = Roll(scope, grad_inputs[0], Neg(scope, shift), axis);
  grad_outputs->push_back(grad_op);
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("Roll", RollGrad);

}  // namespace
}  // namespace ops
}  // namespace machina
