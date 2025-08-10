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

#ifndef MACHINA_CC_GRADIENTS_GRAD_TESTUTIL_H_
#define MACHINA_CC_GRADIENTS_GRAD_TESTUTIL_H_

#include <vector>

#include "absl/status/status.h"
#include "machina/cc/framework/ops.h"
#include "machina/cc/framework/scope.h"

namespace machina {
namespace test {

/// Calls the gradient function registered for 'op', adding gradient operations
/// to the graph associated with 'scope'. Gradient outputs for each 'op' input
/// are returned in 'grad_outputs'.
absl::Status CallGradFunction(const Scope& scope, const Operation& op,
                              const std::vector<Output>& grad_inputs,
                              std::vector<Output>* grad_outputs);

}  // namespace test
}  // namespace machina

#endif  // MACHINA_CC_GRADIENTS_GRAD_TESTUTIL_H_
