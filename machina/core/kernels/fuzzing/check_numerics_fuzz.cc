/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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

#include "machina/cc/framework/scope.h"
#include "machina/cc/ops/array_ops.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/kernels/fuzzing/fuzz_session.h"

namespace machina {
namespace fuzzing {

class FuzzCheckNumerics : public FuzzSession {
  void BuildGraph(const Scope& scope) override {
    auto input =
        machina::ops::Placeholder(scope.WithOpName("input"), DT_FLOAT);
    auto prefix = "Error: ";
    (void)machina::ops::CheckNumerics(scope.WithOpName("output"), input,
                                         prefix);
  }

  void FuzzImpl(const uint8_t* data, size_t size) override {
    size_t ratio = sizeof(float) / sizeof(uint8_t);
    size_t num_floats = size / ratio;
    const float* float_data = reinterpret_cast<const float*>(data);

    Tensor input_tensor(machina::DT_FLOAT,
                        TensorShape({static_cast<int64_t>(num_floats)}));
    auto flat_tensor = input_tensor.flat<float>();
    for (size_t i = 0; i < num_floats; i++) {
      flat_tensor(i) = float_data[i];
    }
    RunInputs({{"input", input_tensor}});
  }
};

STANDARD_TF_FUZZ_FUNCTION(FuzzCheckNumerics);

}  // end namespace fuzzing
}  // end namespace machina
