/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

// Definition for the ops used to swap data in and out of GPU memory.

#include "machina/core/framework/op.h"
#include "machina/core/framework/shape_inference.h"
#include "machina/core/lib/core/status.h"

namespace machina {
namespace {

// The _CopyFromGpuToHost op copies its input tensor to the host. The input must
// reside on GPU. The op itself must be placed on GPU.
REGISTER_OP("_CopyFromGpuToHost")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data != nullptr) {
        c->set_output_handle_shapes_and_types(0, *handle_data);
      }
      return absl::OkStatus();
    })
    .Doc("Copies the input tensor from gpu to the host.");

// The _CopyFromHostToGpu op copies its input tensor from the host to the GPU.
// The input must reside on CPU. The op itself must be placed on GPU.
REGISTER_OP("_CopyFromHostToGpu")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data != nullptr) {
        c->set_output_handle_shapes_and_types(0, *handle_data);
      }
      return absl::OkStatus();
    })
    .Doc("Copies the input tensor from the host to the GPU.");

}  // namespace
}  // namespace machina
