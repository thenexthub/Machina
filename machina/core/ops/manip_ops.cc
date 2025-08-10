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

#include "machina/core/framework/common_shape_fns.h"
#include "machina/core/framework/op.h"
#include "machina/core/framework/shape_inference.h"

namespace machina {

// --------------------------------------------------------------------------
REGISTER_OP("Roll")
    .Input("input: T")
    .Input("shift: Tshift")
    .Input("axis: Taxis")
    .Output("output: T")
    .Attr("T: type")
    .Attr("Tshift: {int32,int64}")
    .Attr("Taxis: {int32,int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // The `input` must be 1-D or higher
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &unused));
      // The `shift` must be scalar or 1-D.
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(1), 1, &unused));
      // The `axis` must be scalar or 1-D.
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(2), 1, &unused));
      // Validate 'shift' is the same shape as axis'.
      TF_RETURN_IF_ERROR(c->Merge(c->input(1), c->input(2), &unused));
      return shape_inference::UnchangedShape(c);
    });

}  // namespace machina
