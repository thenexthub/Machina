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

#ifndef MACHINA_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_PASSES_H_
#define MACHINA_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_PASSES_H_

#include <functional>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain

namespace mlir {
namespace quant {

using OperationToName = std::function<toolchain::StringRef(Operation* op)>;

// Creates an instance pass to import quantization stats to the operations in
// the function. A custom method to get the name from the op is used because
// different dialect ops might have different ways to assign the name.
std::unique_ptr<OperationPass<func::FuncOp>> CreateImportQuantStatsPass(
    OperationToName op_to_name, const std::string& stats_str);

// Creates an instance pass to import quantization stats to the operations in
// the function. A custom method to get the name from the op is used because
// different dialect ops might have different ways to assign the name.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateImportQuantStatsPassForTFControlDialect(const std::string& stats_str);

}  // namespace quant
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_PASSES_H_
