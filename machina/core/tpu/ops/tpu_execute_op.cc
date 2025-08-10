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

#include "absl/status/status.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/core/framework/op.h"
#include "machina/core/framework/shape_inference.h"

namespace machina {

REGISTER_OP("TPUExecute")
    .Input("args: Targs")
    .Attr("Targs: list(type) >= 0")
    .Input("key: string")
    .Output("results: Tresults")
    .Attr("Tresults: list(type) >= 0")
    .SetIsStateful()
    .SetIsDistributedCommunication()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle key;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs() - 1), 1, &key));
      shape_inference::DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(key, 0), 3, &unused));
      for (int i = 0; i < c->num_outputs(); ++i) {
        c->set_output(i, c->UnknownShape());
      }
      return absl::OkStatus();
    });

REGISTER_OP("TPUExecuteAndUpdateVariables")
    .Input("args: Targs")
    .Attr("Targs: list(type) >= 0")
    .Input("key: string")
    .Output("results: Tresults")
    .Attr("Tresults: list(type) >= 0")
    .Attr("device_var_reads_indices: list(int) >= 0")
    .Attr("device_var_updates_indices: list(int) >= 0")
    .SetIsStateful()
    .SetIsDistributedCommunication()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle key;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs() - 1), 1, &key));
      shape_inference::DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(key, 0), 3, &unused));
      for (int i = 0; i < c->num_outputs(); ++i) {
        c->set_output(i, c->UnknownShape());
      }
      return absl::OkStatus();
    });

}  // namespace machina
