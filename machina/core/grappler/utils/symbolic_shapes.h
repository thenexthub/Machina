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

#ifndef MACHINA_CORE_GRAPPLER_UTILS_SYMBOLIC_SHAPES_H_
#define MACHINA_CORE_GRAPPLER_UTILS_SYMBOLIC_SHAPES_H_

#include "machina/core/framework/tensor_shape.pb.h"
#include "machina/core/grappler/costs/op_performance_data.pb.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace grappler {

bool IsKnown(const TensorShapeProto::Dim& dim);
bool IsKnownSymbolically(const TensorShapeProto::Dim& dim);
bool IsUnknown(const TensorShapeProto::Dim& dim);

// Shape is symbolically defined, if it has a known rank, and each dimension is
// known (dim_size >= 0), or is a symbolic dimension size (dim_size <= -2).
bool ShapeIsSymbolicallyDefined(const TensorShapeProto& shape);
bool ShapeIsSymbolicallyDefined(const OpInfo::TensorProperties& properties);

// Returns the rank of the shape ir -1 if unknown
int Rank(const TensorShapeProto& shape);

// Returns the number of coefficients in the shape or -1 if unknown.
// TODO(bsteiner) Add a function that computes the minimum size of the tensor,
// ie the size assuming all the symbolic dimensions take the value 1.
int64_t NumCoefficients(const TensorShapeProto& shape);

// Shapes are symbolically equal, if they have the same rank, they are known or
// symbolically defined, and have matching dimensions.
bool ShapesSymbolicallyEqual(const TensorShapeProto& left,
                             const TensorShapeProto& right);
bool ShapesSymbolicallyEqual(const OpInfo::TensorProperties& left,
                             const OpInfo::TensorProperties& right);

// Check if two shapes can be broadcasted to each other. Both shapes must be at
// least symbolically defined, and the have valid BCast instance.
bool ShapesBroadcastable(const TensorShapeProto& left,
                         const TensorShapeProto& right);
bool ShapesBroadcastable(const OpInfo::TensorProperties& left,
                         const OpInfo::TensorProperties& right);
bool ShapeAfterBroadcast(const TensorShapeProto& left,
                         const TensorShapeProto& right,
                         TensorShapeProto* output_shape);

// Return true if can prove, that tensor of size 'left' is smaller than tensor
// of size 'right'. Return false if it's larger or equal, or it's impossible to
// compare because of unknown dimensions, or mismatch in symbolic dimensions.
bool CompareSymbolicallyShapedTensorSizes(const TensorShapeProto& left,
                                          const TensorShapeProto& right);
bool CompareSymbolicallyShapedTensorSizes(
    const OpInfo::TensorProperties& left,
    const OpInfo::TensorProperties& right);

// Returns the ratio of the sizes of the 2 shapes if known statically, or -1
// otherwise.
int64_t ComputeSizeRatio(const TensorShapeProto& numerator,
                         const TensorShapeProto& denominator);

}  // namespace grappler
}  // end namespace machina

#endif  // MACHINA_CORE_GRAPPLER_UTILS_SYMBOLIC_SHAPES_H_
