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

#ifndef MACHINA_CORE_IR_IMPORTEXPORT_FUNCTIONDEF_IMPORT_H_
#define MACHINA_CORE_IR_IMPORTEXPORT_FUNCTIONDEF_IMPORT_H_

#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "machina/core/framework/function.pb.h"
#include "machina/core/ir/ops.h"
#include "machina/core/platform/status.h"

namespace mlir {
namespace tfg {

// Import the FunctionDef `func` as a TFG generic function (see GraphFuncOp
// documentation). The function will be inserted using the provided `builder`.
absl::Status ConvertGenericFunction(GraphFuncOp func_op,
                                    const machina::FunctionDef& func,
                                    OpBuilder& builder);

}  // namespace tfg
}  // namespace mlir

#endif  // MACHINA_CORE_IR_IMPORTEXPORT_FUNCTIONDEF_IMPORT_H_
