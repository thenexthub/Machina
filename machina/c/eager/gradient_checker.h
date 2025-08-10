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
#ifndef MACHINA_C_EAGER_GRADIENT_CHECKER_H_
#define MACHINA_C_EAGER_GRADIENT_CHECKER_H_

#include <memory>

#include "absl/types/span.h"
#include "machina/c/eager/abstract_tensor_handle.h"
#include "machina/c/eager/unified_api_testutil.h"

namespace machina {
namespace gradients {

/* Returns numerical grad inside `dtheta_approx` given `forward` model and
 * parameter specified by `input_index`.
 *
 * I.e. if y = <output of the forward model> and w = inputs[input_index],
 * this will calculate dy/dw numerically.
 *
 * `use_function` indicates whether to use graph mode(true) or eager(false).
 *
 * `numerical_grad` is the pointer to the AbstractTensorHandle* which will
 * hold the numerical gradient data at the end of the function.
 */
absl::Status CalcNumericalGrad(AbstractContext* ctx, Model forward,
                               absl::Span<AbstractTensorHandle* const> inputs,
                               int input_index, bool use_function,
                               AbstractTensorHandle** numerical_grad);

}  // namespace gradients
}  // namespace machina

#endif  // MACHINA_C_EAGER_GRADIENT_CHECKER_H_
