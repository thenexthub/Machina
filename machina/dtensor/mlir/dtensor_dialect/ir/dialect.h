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

#ifndef MACHINA_DTENSOR_MLIR_DTENSOR_DIALECT_IR_DIALECT_H_
#define MACHINA_DTENSOR_MLIR_DTENSOR_DIALECT_IR_DIALECT_H_

#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Dialect.h"  // part of Codira Toolchain

// Dialect main class is defined in ODS, we include it here. The
// constructor and the printing/parsing of dialect types are manually
// implemented (see ops.cpp).
#include "machina/dtensor/mlir/dtensor_dialect/ir/dialect.h.inc"

namespace mlir {
namespace dtensor {

//===----------------------------------------------------------------------===//
// DTENSOR dialect types.
//===----------------------------------------------------------------------===//

}  // namespace dtensor
}  // namespace mlir

#endif  // MACHINA_DTENSOR_MLIR_DTENSOR_DIALECT_IR_DIALECT_H_
