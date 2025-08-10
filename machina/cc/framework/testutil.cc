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

#include "machina/cc/framework/testutil.h"

#include <utility>

#include "machina/cc/client/client_session.h"
#include "machina/core/framework/tensor_testutil.h"
#include "machina/core/graph/default_device.h"

namespace machina {
namespace test {

void GetTensors(const Scope& scope, OutputList tensors,
                std::vector<Tensor>* out) {
  ClientSession session(scope);
  TF_CHECK_OK(session.Run(tensors, out));
}

void GetTensor(const Scope& scope, Output tensor, Tensor* out) {
  std::vector<Tensor> outputs;
  GetTensors(scope, {std::move(tensor)}, &outputs);
  *out = outputs[0];
}

void GetTensors(const Scope& scope, const std::vector<Output>& assign_vars,
                const OutputList& tensors, std::vector<Tensor>* out) {
  ClientSession session(scope);
  TF_CHECK_OK(session.Run(assign_vars, nullptr));
  TF_CHECK_OK(session.Run(tensors, out));
}

void GetTensor(const Scope& scope, const std::vector<Output>& assign_vars,
               Output tensor, Tensor* out) {
  std::vector<Tensor> outputs;
  GetTensors(scope, assign_vars, {std::move(tensor)}, &outputs);
  *out = outputs[0];
}

}  // end namespace test
}  // end namespace machina
