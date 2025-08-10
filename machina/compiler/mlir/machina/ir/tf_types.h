/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

// This file defines the types used in the standard MLIR TensorFlow dialect.

#ifndef MACHINA_COMPILER_MLIR_MACHINA_IR_TF_TYPES_H_
#define MACHINA_COMPILER_MLIR_MACHINA_IR_TF_TYPES_H_

#include "machina/core/ir/types/dialect.h"

namespace mlir {
namespace TF {

// This all moved under machina/core/ir/types and these using declaration are
// to help with the transition.

using ::mlir::tf_type::AreCastCompatible;          // NOLINT
using ::mlir::tf_type::ArraysAreCastCompatible;    // NOLINT
using ::mlir::tf_type::BroadcastCompatible;        // NOLINT
using ::mlir::tf_type::DropRefType;                // NOLINT
using ::mlir::tf_type::filter_resources;           // NOLINT
using ::mlir::tf_type::GetCastCompatibleType;      // NOLINT
using ::mlir::tf_type::HasCompatibleElementTypes;  // NOLINT
using ::mlir::tf_type::IsValidTFTensorType;        // NOLINT
using ::mlir::tf_type::OperandShapeIterator;       // NOLINT
using ::mlir::tf_type::ResourceType;               // NOLINT
using ::mlir::tf_type::ResultShapeIterator;        // NOLINT
using ::mlir::tf_type::ResultShapeRange;           // NOLINT
using ::mlir::tf_type::StringType;                 // NOLINT
using ::mlir::tf_type::TensorFlowRefType;          // NOLINT
using ::mlir::tf_type::TensorFlowType;             // NOLINT
using ::mlir::tf_type::TensorFlowTypeWithSubtype;  // NOLINT
using ::mlir::tf_type::VariantType;                // NOLINT

#define HANDLE_TF_TYPE(tftype, enumerant, name) \
  using tftype##Type = mlir::tf_type::tftype##Type;
#include "machina/compiler/mlir/machina/ir/tf_types.def"


}  // end namespace TF
}  // end namespace mlir

#endif  // MACHINA_COMPILER_MLIR_MACHINA_IR_TF_TYPES_H_
