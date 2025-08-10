/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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
#ifndef MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_CC_CONST_OP_SIZE_H_
#define MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_CC_CONST_OP_SIZE_H_

#include <cstdint>

#include "machina/compiler/mlir/machina/ir/tf_ops.h"

namespace mlir {
namespace quant {

// Returns the size in bytes of the underlying data of `const_op`. If the
// underlying type's size cannot be determined, it assumes 4 bytes per element.
int64_t GetSizeInBytes(TF::ConstOp const_op);

}  // namespace quant
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_CC_CONST_OP_SIZE_H_
