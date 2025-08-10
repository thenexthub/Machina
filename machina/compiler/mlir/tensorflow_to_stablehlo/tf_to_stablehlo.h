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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_TO_STABLEHLO_TF_TO_STABLEHLO_H_
#define MACHINA_COMPILER_MLIR_MACHINA_TO_STABLEHLO_TF_TO_STABLEHLO_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain

namespace mlir {

// Converts a TensorFlow model (either from a SavedModel or an MLIR module) to a
// StableHLO MLIR module.
//
// Args:
//  input_path: The path to the input TensorFlow SavedModel or MLIR module.
//  context: The MLIR context to use for parsing or creating the MLIR module.
//  exported_model_signatures: List of exported model signatures (strings) to
//    convert.
//  tag_names: List of tag names (strings) used for loading SavedModel.
//    Ignored for MLIR input.
//  input_arg_shapes_str:  A string representation of input argument shapes for
//    'main' entry-point, separating tensors with ':', dimension with ',', and
//    using '?' for unknown sizes. For example, 'input-arg-shapes=1,2::1,?'
//    expresses argument shapes [1,2], [] and [1,?].
//  is_input_mlir_module: If true, `input_path` is treated as an MLIR
//    module instead of a SavedModel.
//
// Returns:
//   An absl::StatusOr containing the converted StableHLO MLIR module on
//   success, or an absl::Status with an error message on failure.
absl::StatusOr<OwningOpRef<ModuleOp>> TfToStablehlo(
    absl::string_view input_path, MLIRContext* context,
    const std::vector<std::string>& exported_model_signatures,
    const std::vector<std::string>& tag_names,
    absl::string_view input_arg_shapes_str, bool is_input_mlir_module);

}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_MACHINA_TO_STABLEHLO_TF_TO_STABLEHLO_H_
