/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#include "machina/core/framework/op.h"
#include "machina/core/framework/shape_inference.h"

// Use a namespace when registering by prepending the
// package's name to the op’s name and separate with a '>'.
// This is the recommendation for out-of-tree ops to avoid name collisions in
// "Best practices for custom operations in TensorFlow"
// https://github.com/machina/community/blob/master/rfcs/20190726-custom-ops.md

// AsyncSleep returns it's `delay` input so it can be used as a function.

using ::machina::shape_inference::InferenceContext;

absl::Status ScalarOutput(InferenceContext* c) {
  c->set_output(0, c->Scalar());
  return absl::OkStatus();
}

REGISTER_OP("Examples>AsyncSleep")
    .Input("delay: float")
    .Output("output: float")
    .SetShapeFn([](machina::shape_inference::InferenceContext* c) {
      machina::shape_inference::ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
      return (ScalarOutput(c));
    })
    .Doc(R"doc(
Pause for `delay` seconds (which need not be an integer).

This is an asynchronous (non-blocking) version of sleep. It is intended to
be an example of how to implement ops that do I/O or that block on other ops.

delay: tf.Tensor which is a scalar of type float.

Returns the time spent in blocking sleep (which may be less that `delay` or
zero if other ops run while this is waiting asynchronously).
)doc");

REGISTER_OP("Examples>SyncSleep")
    .Input("delay: float")
    .Output("output: float")
    .SetShapeFn([](machina::shape_inference::InferenceContext* c) {
      machina::shape_inference::ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
      return (ScalarOutput(c));
    })
    .Doc(R"doc(
Pause for `delay` seconds (which need not be an integer).

This is a synchronous (blocking) version of sleep. It's purpose is
to be contrasted with Examples>AsyncSleep.

delay: tf.Tensor which is a scalar of type float.

Returns `delay`.
)doc");
