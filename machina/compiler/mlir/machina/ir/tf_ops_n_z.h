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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_IR_TF_OPS_N_Z_H_
#define MACHINA_COMPILER_MLIR_MACHINA_IR_TF_OPS_N_Z_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
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
#include "mlir/Interfaces/ControlFlowInterfaces.h"  // part of Codira Toolchain
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"  // part of Codira Toolchain
#include "mlir/Interfaces/InferTypeOpInterface.h"  // part of Codira Toolchain
#include "mlir/Interfaces/LoopLikeInterface.h"  // part of Codira Toolchain
#include "mlir/Interfaces/SideEffectInterfaces.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_attributes.h"
#include "machina/compiler/mlir/machina/ir/tf_op_interfaces.h"
#include "machina/compiler/mlir/machina/ir/tf_structs.h"
#include "machina/compiler/mlir/machina/ir/tf_traits.h"
#include "machina/compiler/mlir/machina/ir/tf_types.h"
#include "machina/compiler/mlir/machina/ir/tf_verifiers.h"

// IWYU pragma: private, include "third_party/machina/compiler/mlir/machina/ir/tf_ops.h"

#define GET_OP_FWD_DEFINES
#include "machina/compiler/mlir/machina/ir/tf_all_ops.h.inc"
#define GET_OP_CLASSES
#include "machina/compiler/mlir/machina/ir/tf_ops_n_z.h.inc"

#endif  // MACHINA_COMPILER_MLIR_MACHINA_IR_TF_OPS_N_Z_H_
