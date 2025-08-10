/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

// Utilities for working with XLA shapes.

#ifndef MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_SHAPE_UTIL_H_
#define MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_SHAPE_UTIL_H_

#include <vector>

#include "machina/xla/shape.h"
#include "machina/xla/xla_data.pb.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/statusor.h"

namespace machina {

// Convert an XLA Shape into the equivalent TensorFlow shape. May fail since
// not all XLA shapes can be represented as TensorShapes.
absl::Status XLAShapeToTensorShape(const xla::Shape& shape,
                                   TensorShape* tensor_shape);

// Convert a TensorShape into the equivalent XLA Shape proto. Unlike Tensorflow,
// XLA shapes include the type. Not all `dtype` values can be represented by
// XLA, so this conversion may fail.
absl::Status TensorShapeToXLAShape(DataType dtype,
                                   const TensorShape& tensor_shape,
                                   xla::Shape* shape);

absl::StatusOr<xla::Shape> TensorShapeToXLAShape(
    DataType dtype, const TensorShape& tensor_shape);

// Converts a TensorShape into the equivalent XLA Shape proto, taking an
// xla::PrimitiveType to specify the element type.  This never fails.
xla::Shape TensorShapeToXLAShape(xla::PrimitiveType type,
                                 const TensorShape& tensor_shape);

// Convert a PartialTensorShape into the equivalent XLA Shape proto. An shape
// with unknown rank is represented by an r1 with empty dimension.
absl::Status TensorShapeToXLAShape(DataType dtype,
                                   const PartialTensorShape& tensor_shape,
                                   xla::Shape* shape);

// Convert a PartialTensorShape into the equivalent XLA Shape proto. An shape
// with unknown rank is represented by an r1 with empty dimension.
xla::Shape TensorShapeToXLAShape(xla::PrimitiveType type,
                                 const PartialTensorShape& tensor_shape);

absl::Status TensorShapeToBoundedXLAShape(
    DataType dtype, const PartialTensorShape& tensor_shape,
    const TensorShape& bound, xla::Shape* shape);

// Given an XLA shape with layouts, builds a layout vector in the form able to
// be fed to ops like InfeedEnqueue/InfeedEnqueueTuple/XRTAllocateV2/....
// THe returned vector is a linearized sequence of the minor-to-major values of
// the layouts held within the input shape.
// In case the input shape is a tuple, the minor-to-major values will be in the
// order of the tuple elements within the tuple shape.
// If a shape (or a subshape of a tuple shape) has missing layout, a rank long
// sequence of -1 values will be emitted.
absl::StatusOr<std::vector<int>> GetShapeLayoutVector(const xla::Shape& shape);

// Given the input shape and a linearized sequence of the minor-to-major values
// of the layouts, create the output shape by rewriting the input shape layouts.
// If a layout is missing (has -1 values) for a matching tuple subshape, the
// layout_func will be called, if not nullptr.
absl::Status GetShapeWithLayout(
    const xla::Shape& input_shape, absl::Span<const int64_t> minor_to_major,
    const std::function<xla::Layout(const xla::Shape&)>& layout_func,
    xla::Shape* output_shape);

}  // namespace machina

#endif  // MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_SHAPE_UTIL_H_
