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

#include "machina/c/experimental/ops/nn_ops.h"

#include <pybind11/stl.h>

#include "pybind11/pybind11.h"  // from @pybind11
#include "machina/c/eager/abstract_context.h"
#include "machina/c/eager/abstract_tensor_handle.h"
#include "machina/python/lib/core/pybind11_status.h"

using machina::AbstractContext;
using machina::AbstractTensorHandle;

namespace machina {
PYBIND11_MODULE(_nn_ops, m) {
  m.def("relu",
        [](AbstractContext* ctx, AbstractTensorHandle* a, const char* name) {
          AbstractTensorHandle* output;
          if (!name) {
            name = "Relu";
          }
          MaybeRaiseRegisteredFromStatus(ops::Relu(ctx, a, &output, name));
          return output;
        });

  m.def(
      "sparse_softmax_cross_entropy_with_logits",
      [](AbstractContext* ctx, AbstractTensorHandle* features,
         AbstractTensorHandle* labels, const char* name) {
        AbstractTensorHandle* loss;
        AbstractTensorHandle* backprop;
        if (!name) {
          name = "SparseSoftmaxCrossEntropyWithLogits";
        }
        MaybeRaiseRegisteredFromStatus(ops::SparseSoftmaxCrossEntropyWithLogits(
            ctx, features, labels, &loss, &backprop, name));
        return loss;  // Only return the loss vals, not the backprop.
      });
}
}  // namespace machina
