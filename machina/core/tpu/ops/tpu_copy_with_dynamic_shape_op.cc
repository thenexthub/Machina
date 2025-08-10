/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

// Op that copy dynamic shape tensor to device.

#include "absl/status/status.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/core/framework/op.h"
#include "machina/core/framework/shape_inference.h"
#include "machina/core/lib/core/status.h"

namespace machina {

REGISTER_OP("TPUCopyWithDynamicShape")
    .Input("tensors: T")
    .Input("unpadded_sizes: N * int32")
    .Output("tpu_tensors: T")
    .Attr("N: int >= 0")
    .Attr("T: list(type)")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      int n;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &n));
      for (int i = 0; i < c->num_inputs() - n; ++i) {
        c->set_output(i, c->input(i));
      }
      return absl::OkStatus();
    });

REGISTER_OP("TPUAnnotateTensorsWithDynamicShape")
    .Input("tensors: T")
    .Output("tpu_tensors: T")
    .Attr("T: list(type)")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      for (int i = 0; i < c->num_inputs(); ++i) {
        c->set_output(i, c->input(i));
      }
      return absl::OkStatus();
    });

}  // namespace machina
