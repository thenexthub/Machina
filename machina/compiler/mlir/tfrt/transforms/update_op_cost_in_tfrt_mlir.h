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
#ifndef MACHINA_COMPILER_MLIR_TFRT_TRANSFORMS_UPDATE_OP_COST_IN_TFRT_MLIR_H_
#define MACHINA_COMPILER_MLIR_TFRT_TRANSFORMS_UPDATE_OP_COST_IN_TFRT_MLIR_H_

#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "machina/core/tfrt/fallback/cost_recorder.h"

namespace machina {
namespace tfrt_compiler {

// Updates the existing costs for all the fallback ops with the records in
// `cost_recorder`.
void UpdateOpCostInTfrtMlir(mlir::ModuleOp op,
                            const tfrt_stub::CostRecorder& cost_recorder);

}  // namespace tfrt_compiler
}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TFRT_TRANSFORMS_UPDATE_OP_COST_IN_TFRT_MLIR_H_
