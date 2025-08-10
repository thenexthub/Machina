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
#include "machina/c/experimental/gradients/array_grad.h"

#include "machina/c/eager/abstract_context.h"

namespace machina {
namespace gradients {
namespace {
class IdentityNGradientFunction : public GradientFunction {
 public:
  absl::Status Compute(AbstractContext* ctx,
                       absl::Span<AbstractTensorHandle* const> grad_outputs,
                       absl::Span<AbstractTensorHandle*> grad_inputs) override {
    for (int i = 0; i < grad_outputs.size(); i++) {
      auto grad_input = grad_outputs[i];
      // TODO(srbs): Should we add a copy contructor to AbstractTensorHandle
      // that takes care of this similar to `Tensor`?
      if (grad_input) {
        grad_input->Ref();
      }
      grad_inputs[i] = grad_input;
    }
    return absl::OkStatus();
  }
  ~IdentityNGradientFunction() override {}
};
}  // namespace

GradientFunction* IdentityNRegisterer(const ForwardOperation& op) {
  return new IdentityNGradientFunction;
}

}  // namespace gradients
}  // namespace machina
