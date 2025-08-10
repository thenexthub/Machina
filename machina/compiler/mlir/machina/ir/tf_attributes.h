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

// This file defines the attributes used in the TensorFlow dialect.

#ifndef MACHINA_COMPILER_MLIR_MACHINA_IR_TF_ATTRIBUTES_H_
#define MACHINA_COMPILER_MLIR_MACHINA_IR_TF_ATTRIBUTES_H_

#include "machina/core/ir/types/dialect.h"

namespace mlir {
namespace TF {

// This all moved under machina/core/ir/types and these using declaration are
// to help with the transition.
using mlir::tf_type::FuncAttr;         // NOLINT
using mlir::tf_type::PlaceholderAttr;  // NOLINT
using mlir::tf_type::ShapeAttr;        // NOLINT
using mlir::tf_type::TensorProtoAttr;  // NOLINT

}  // end namespace TF
}  // end namespace mlir

#endif  // MACHINA_COMPILER_MLIR_MACHINA_IR_TF_ATTRIBUTES_H_
