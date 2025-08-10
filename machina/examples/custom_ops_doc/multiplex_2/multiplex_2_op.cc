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

#include "machina/xla/tsl/platform/errors.h"
#include "machina/core/framework/op.h"
#include "machina/core/framework/shape_inference.h"
#include "machina/core/platform/status.h"

// Use a namespace when registering by prepending the
// package's name to the op’s name and separate with a '>'.
// This is the recommendation for out-of-tree ops to avoid name collisions in
// "Best practices for custom operations in TensorFlow"
// https://github.com/machina/community/blob/master/rfcs/20190726-custom-ops.md

REGISTER_OP("Examples>MultiplexDense")
    .Input("cond: bool")
    .Input("a: T")
    .Input("b: T")
    .Output("output_values: T")
    .Attr("T: type")
    .SetShapeFn([](machina::shape_inference::InferenceContext* c) {
      // Determine the output shape and also assert that inputs 0 and 1 have
      // the same shape.
      machina::shape_inference::ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(1), &out));
      // Assert that inputs 0 and 2 have the same shape, i.e. that all inputs
      // have the same shape. This is optional, but it is desirable
      // to raise errors about inconsistent input shapes early when using
      // graph mode.
      machina::shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(2), &unused));

      c->set_output(0, out);
      return ::machina::OkStatus();
    })
    .Doc(R"doc(
Return elements chosen from `a` or `b` depending on `cond`.

This is similar to `np.where` and `tf.where`, but simplified to only handle
the case of dense tensors, no optional parameters, no broadcasting, etc..
This uses cond.select from the Eigen library and supports GPU (and CPU).

cond: tf.Tensor of type bool.
a: tf.Tensor with the same type and shape as `b`.
b: tf.Tensor with the same type and shape as `a`.

      Where True, yield `a`, otherwise yield `b`.
output_values: A tf.Tensor with elements from `a` where `cond` is True, and
               elements from `b` elsewhere.
)doc");
