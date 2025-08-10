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

#include <cstdint>

#include "absl/status/status.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/core/framework/op.h"
#include "machina/core/framework/shape_inference.h"

namespace machina {

using shape_inference::ShapeHandle;

REGISTER_OP("KthOrderStatistic")
    .Input("input: float32")
    .Output("output: float32")
    .Attr("k: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));

      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->Subshape(input, 0, -1, &s));
      c->set_output(0, s);
      return absl::OkStatus();
    });

REGISTER_OP("TopKUnique")
    .Input("input: float32")
    .Output("topk: float32")
    .Output("topk_indices: int32")
    .Attr("k: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));

      int32_t k;
      TF_RETURN_IF_ERROR(c->GetAttr("k", &k));

      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->ReplaceDim(input, 1, c->MakeDim(k), &s));
      c->set_output(0, s);
      c->set_output(1, s);
      return absl::OkStatus();
    });

REGISTER_OP("MakeUnique")
    .Input("input: float32")
    .Output("output: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
      c->set_output(0, input);
      return absl::OkStatus();
    });

REGISTER_OP("TopKWithUnique")
    .Input("input: float32")
    .Output("topk: float32")
    .Output("topk_indices: int32")
    .Attr("k: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));

      int32_t k;
      TF_RETURN_IF_ERROR(c->GetAttr("k", &k));

      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->ReplaceDim(input, 1, c->MakeDim(k), &s));
      c->set_output(0, s);
      c->set_output(1, s);
      return absl::OkStatus();
    });
}  // namespace machina
