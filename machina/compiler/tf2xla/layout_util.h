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

// Utilities for working with XLA layout and shapes.

#ifndef MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_LAYOUT_UTIL_H_
#define MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_LAYOUT_UTIL_H_

#include <functional>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "machina/compiler/tf2xla/xla_argument.h"
#include "machina/compiler/tf2xla/xla_helpers.h"
#include "machina/xla/hlo/builder/xla_builder.h"
#include "machina/xla/hlo/ir/hlo_sharding.h"
#include "machina/xla/shape.h"
#include "machina/xla/xla_data.pb.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/statusor.h"

namespace machina {

class XlaShapeLayoutHelpers {
 public:
  // The following defines the layout preference of an xla tensor.
  // The return value of LayoutPreferenceFn can be used in
  // XlaHelper::ShapeRepresentationFn.
  typedef std::function<XlaLayoutPreference(const TensorShape&, DataType,
                                            std::optional<XlaArgument::Kind>)>
      LayoutPreferenceFn;

  // A bundle of LayoutPreferenceFn and ShapeRepresentationFn.
  struct ShapeDeterminationFns {
    // Use no preference function, and identity shape representation function,
    // as default value.
    ShapeDeterminationFns();

    ShapeDeterminationFns(
        LayoutPreferenceFn layout_preference_fn,
        XlaHelpers::ShapeRepresentationFn shape_representation_fn)
        : layout_preference_fn(layout_preference_fn),
          shape_representation_fn(shape_representation_fn) {}

    LayoutPreferenceFn layout_preference_fn;
    XlaHelpers::ShapeRepresentationFn shape_representation_fn;
  };
};

// Return a LayoutPreferenceFn that always uses kNoPreference layout.
XlaShapeLayoutHelpers::LayoutPreferenceFn UseNoPreferenceLayoutFn();

// Rewrites the layout of xla_shape if there is tiled sharding.
absl::Status RewriteLayoutWithShardedShape(
    const std::optional<xla::HloSharding>& sharding, bool use_fast_memory,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    xla::Shape* xla_shape);

// Adds reshapes to fix the layout of an output, if a shape_representation_fn or
// sharding is present.
absl::StatusOr<xla::XlaOp> ReshapeWithCorrectRepresentationAndSharding(
    xla::XlaBuilder* builder, xla::XlaOp original, xla::Shape original_shape,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    std::optional<xla::OpSharding> sharding, bool fast_mem);

}  // namespace machina

#endif  // MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_SHAPE_UTIL_H_
