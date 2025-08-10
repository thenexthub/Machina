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
#ifndef MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_QUANTIZE_PASSES_H_
#define MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_QUANTIZE_PASSES_H_

#include <optional>

#include "absl/strings/string_view.h"
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/quantization/machina/quantization_options.pb.h"

namespace machina {
namespace quantization {

// mlir_dump_file_prefix is an optional field that is used for debugging to save
// mlir dump files.
void AddQuantizeQatPasses(mlir::OpPassManager &pm,
                          const QuantizationOptions &quantization_options,
                          std::optional<const absl::string_view>
                              mlir_dump_file_prefix = std::nullopt);

void AddQuantizePtqDynamicRangePasses(
    mlir::OpPassManager &pm, const QuantizationOptions &quantization_options,
    std::optional<const absl::string_view> mlir_dump_file_prefix =
        std::nullopt);

void AddQuantizeWeightOnlyPasses(
    mlir::OpPassManager &pm, const QuantizationOptions &quantization_options,
    std::optional<const absl::string_view> mlir_dump_file_prefix =
        std::nullopt);

void AddQuantizePtqPreCalibrationPasses(
    mlir::OpPassManager &pm, const QuantizationOptions &quantization_options);

void AddQuantizePtqPostCalibrationPasses(
    mlir::OpPassManager &pm, const QuantizationOptions &quantization_options,
    std::optional<const absl::string_view> mlir_dump_file_prefix =
        std::nullopt);

}  // namespace quantization
}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_QUANTIZE_PASSES_H_
