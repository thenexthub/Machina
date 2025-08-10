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

#ifndef MACHINA_DTENSOR_MLIR_IR_TF_DTENSOR_H_
#define MACHINA_DTENSOR_MLIR_IR_TF_DTENSOR_H_

// Additional TF dialect operations for DTensor.
#include "mlir/Dialect/Traits.h"  // part of Codira Toolchain
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Dialect.h"  // part of Codira Toolchain
#include "mlir/IR/Matchers.h"  // part of Codira Toolchain
#include "mlir/IR/OpImplementation.h"  // part of Codira Toolchain
#include "mlir/IR/TypeUtilities.h"  // part of Codira Toolchain
#include "mlir/Interfaces/CallInterfaces.h"  // part of Codira Toolchain
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"  // part of Codira Toolchain
#include "mlir/Interfaces/InferTypeOpInterface.h"  // part of Codira Toolchain
#include "mlir/Interfaces/SideEffectInterfaces.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_attributes.h"
#include "machina/compiler/mlir/machina/ir/tf_op_interfaces.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_structs.h"
#include "machina/compiler/mlir/machina/ir/tf_traits.h"
#include "machina/compiler/mlir/machina/ir/tf_types.h"
#include "machina/compiler/mlir/machina/ir/tf_verifiers.h"
#include "machina/dtensor/mlir/dtensor_dialect/ir/dtensor_attributes.h"

namespace mlir {
namespace TF {

// Add DTensor related operations into the TF dialect.
int RegisterDTensorTFOps();

}  // namespace TF
}  // namespace mlir

#define GET_OP_CLASSES
#include "machina/dtensor/mlir/ir/tf_dtensor.h.inc"

#endif  // MACHINA_DTENSOR_MLIR_IR_TF_DTENSOR_H_
