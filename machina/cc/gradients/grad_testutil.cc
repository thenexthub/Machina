/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#include "machina/cc/gradients/grad_testutil.h"

#include <vector>

#include "absl/status/status.h"
#include "machina/cc/framework/grad_op_registry.h"

namespace machina {
namespace test {

absl::Status CallGradFunction(const Scope& scope, const Operation& op,
                              const std::vector<Output>& grad_inputs,
                              std::vector<Output>* grad_outputs) {
  ops::GradFunc grad_fn;
  TF_RETURN_IF_ERROR(ops::GradOpRegistry::Global()->Lookup(
      op.node()->type_string(), &grad_fn));
  TF_RETURN_IF_ERROR(grad_fn(scope, op, grad_inputs, grad_outputs));
  TF_RETURN_IF_ERROR(scope.status());
  return absl::OkStatus();
}

}  // end namespace test
}  // end namespace machina
