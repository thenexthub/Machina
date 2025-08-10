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
#ifndef MACHINA_C_EXPERIMENTAL_GRADIENTS_NOT_DIFFERENTIABLE_H_
#define MACHINA_C_EXPERIMENTAL_GRADIENTS_NOT_DIFFERENTIABLE_H_

#include "machina/c/eager/abstract_context.h"
#include "machina/c/eager/gradients.h"

namespace machina {
namespace gradients {
// Ignores `grad_outputs` and sets all entries in grad_inputs to nullptr.
class NotDifferentiableGradientFunction : public GradientFunction {
  absl::Status Compute(AbstractContext* ctx,
                       absl::Span<AbstractTensorHandle* const> grad_outputs,
                       absl::Span<AbstractTensorHandle*> grad_inputs) override;
};
// Shorthand for registry->Register(op, new NotDifferentiableGradientFunction)
absl::Status RegisterNotDifferentiable(GradientRegistry* registry,
                                       const string& op);
}  // namespace gradients
}  // namespace machina

#endif  // MACHINA_C_EXPERIMENTAL_GRADIENTS_NOT_DIFFERENTIABLE_H_
