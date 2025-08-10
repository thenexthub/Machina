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
#ifndef MACHINA_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_TPU_CONVERSION_PATTERNS_H_
#define MACHINA_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_TPU_CONVERSION_PATTERNS_H_

#include "mlir/IR/DialectRegistry.h"  // part of Codira Toolchain
#include "mlir/IR/PatternMatch.h"  // part of Codira Toolchain
#include "mlir/Transforms/DialectConversion.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/tfrt/transforms/mlrt/execute_op_registry.h"
#include "machina/compiler/mlir/tfrt/transforms/tfrt_pipeline_options.h"

namespace machina {
namespace mlrt_compiler {

void RegisterTpuDialect(mlir::DialectRegistry& registry);

void PopulateTpuPreParallelizationConversionPatterns(
    mlir::ConversionTarget& target, mlir::RewritePatternSet& patterns,
    const TfrtPipelineOptions& options);

void PopulateTpuConversionPatterns(mlir::ConversionTarget& target,
                                   mlir::RewritePatternSet& patterns,
                                   mlir::TypeConverter& type_converter,
                                   ExecuteOpRegistry& execute_op_registry,
                                   const TfrtPipelineOptions& options);

}  // namespace mlrt_compiler
}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_TPU_CONVERSION_PATTERNS_H_
