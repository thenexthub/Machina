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

#ifndef MACHINA_CC_FRAMEWORK_TESTUTIL_H_
#define MACHINA_CC_FRAMEWORK_TESTUTIL_H_

#include <vector>

#include "machina/cc/framework/ops.h"
#include "machina/cc/framework/scope.h"

namespace machina {
namespace test {

/// Computes the outputs listed in 'tensors', returns the tensors in 'out'.
void GetTensors(const Scope& scope, OutputList tensors,
                std::vector<Tensor>* out);

// Computes the outputs listed in 'tensors', returns the tensors in 'out'.
// assign_vars are extra outputs that should be run
// e.g. to assign values to variables.
void GetTensors(const Scope& scope, const std::vector<Output>& assign_vars,
                const OutputList& tensors, std::vector<Tensor>* out);

/// Computes the output 'tensor', returning the resulting tensor in 'out'.
void GetTensor(const Scope& scope, Output tensor, Tensor* out);

// Computes the output 'tensor', returning the resulting tensor in 'out'.
// assign_vars are extra outputs that should be run
// e.g. to assign values to variables.
void GetTensor(const Scope& scope, const std::vector<Output>& assign_vars,
               Output tensor, Tensor* out);

}  // namespace test
}  // namespace machina

#endif  // MACHINA_CC_FRAMEWORK_TESTUTIL_H_
