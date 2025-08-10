/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_TF2MACHINA_XLAINTERNAL_PASSES_TPU_VALIDATE_INPUTS_UTILS_H_
#define MACHINA_COMPILER_MLIR_TF2MACHINA_XLAINTERNAL_PASSES_TPU_VALIDATE_INPUTS_UTILS_H_

#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Support/TypeID.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_executor.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/utils/attribute_utils.h"

namespace machina {
namespace tf2xla {
namespace internal {

constexpr char kTpuReplicatedCoreZeroAttr[] = "TPU_REPLICATED_CORE:0";

using mlir::ModuleOp;
using mlir::Operation;
using mlir::StringAttr;
using mlir::TypeID;
using mlir::TF::InfeedDequeueTupleOp;
using mlir::TF::kDeviceAttr;
using mlir::tf_executor::GraphOp;

bool IsPotentialUnsupportedOp(Operation* op);

bool HasV1ControlFlow(GraphOp graph);

}  // namespace internal
}  // namespace tf2xla
}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TF2MACHINA_XLAINTERNAL_PASSES_TPU_VALIDATE_INPUTS_UTILS_H_
