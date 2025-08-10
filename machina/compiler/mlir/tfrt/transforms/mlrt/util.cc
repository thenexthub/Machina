/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#include "machina/compiler/mlir/tfrt/transforms/mlrt/util.h"

#include "toolchain/Support/Casting.h"
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/host_runtime/tfrt_ops.h.inc"
#include "machina/compiler/mlir/machina/ir/tf_dialect.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_ops_a_m.h.inc"
#include "machina/compiler/mlir/machina/ir/tf_ops_n_z.h.inc"

namespace machina {
namespace mlrt_compiler {

bool UseFallback(mlir::Operation *op) {
  if (!toolchain::isa<mlir::TF::TensorFlowDialect>(op->getDialect())) return false;

  // TODO(b/173017701): have a centralized place to hold the information
  // whether a TF op should be lowered to FallbackExecute op.
  // TODO(b/319045348): Define trait to reflect that IfrtLoadVariableOp has no
  // TF kernels so that we don't need to check every op here.
  return !toolchain::isa<
      mlir::TF::_TfrtSetResourceOp, mlir::TF::_TfrtGetResourceOp,
      mlir::TF::BatchFunctionOp, mlir::TF::CaseOp,
      mlir::TF::IfrtRestoreVariableOp, mlir::TF::StatefulPartitionedCallOp,
      mlir::TF::PartitionedCallOp, mlir::TF::LegacyCallOp, mlir::TF::IfOp,
      mlir::TF::WhileOp, mlir::TF::TPUCompileMlirAndExecuteOp>(op);
}

}  // namespace mlrt_compiler
}  // namespace machina
