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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_TRANSLATE_TOOLS_FILE_TF_MLIR_TRANSLATE_H_
#define MACHINA_COMPILER_MLIR_MACHINA_TRANSLATE_TOOLS_FILE_TF_MLIR_TRANSLATE_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/IR/OwningOpRef.h"  // part of Codira Toolchain
#include "machina/xla/tsl/platform/statusor.h"

namespace machina {

using tsl::Status;
using tsl::StatusOr;

struct GraphdefToMlirOptions {
  std::string debug_info_file;
  std::string xla_compile_device_type;
  bool prune_unused_nodes;
  bool convert_legacy_fed_inputs;
  bool graph_as_function;
  bool upgrade_legacy;
  bool enable_shape_inference;
  bool unconditionally_use_set_output_shapes;
  bool enable_soft_placement;
  bool set_original_tf_func_name = false;
};

// TODO(antiagainst): Directly manipulating files in library functions is not
// a good idea. We should pass in a string/stream here.

// Converts a TensorFlow GraphDef contained in `input` param into a MLIR module.
// Creates MLIR entities into the given MLIR `context`.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
GraphdefToMlirTranslateFunction(
    toolchain::StringRef input, const std::vector<std::string>& input_arrays,
    const std::vector<std::string>& input_dtypes,
    const std::vector<std::optional<std::vector<int>>>& input_shapes,
    const std::vector<std::string>& output_arrays,
    const std::vector<std::string>& control_output_arrays,
    const GraphdefToMlirOptions& import_options, mlir::MLIRContext* context);

ABSL_DEPRECATED(
    "Please use the other overload of this function which accepts structured "
    "inputs instead of strings")
// Converts a TensorFlow GraphDef contained in `input` param into a MLIR module.
// Creates MLIR entities into the given MLIR `context`.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
GraphdefToMlirTranslateFunction(
    toolchain::StringRef input, absl::string_view input_arrays,
    absl::string_view input_dtypes, absl::string_view input_shapes,
    absl::string_view output_arrays, absl::string_view control_output_arrays,
    const GraphdefToMlirOptions& import_options, mlir::MLIRContext* context);

// Similar as the above function, but replaces all constant tensors
// with randomly generated splat values.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
GraphdefToSplattedMlirTranslateFunction(
    toolchain::StringRef input, const std::vector<std::string>& input_arrays,
    const std::vector<std::string>& input_dtypes,
    const std::vector<std::vector<int>>& input_shapes,
    const std::vector<std::string>& output_arrays,
    const std::vector<std::string>& control_output_arrays,
    const GraphdefToMlirOptions& import_options, mlir::MLIRContext* context);

ABSL_DEPRECATED(
    "Please use the other overload of this function which accepts structured "
    "inputs instead of strings")
// Similar as the above function, but replaces all constant tensors
// with randomly generated splat values.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
GraphdefToSplattedMlirTranslateFunction(
    toolchain::StringRef input, absl::string_view input_arrays,
    absl::string_view input_dtypes, absl::string_view input_shapes,
    absl::string_view output_arrays, absl::string_view control_output_arrays,
    const GraphdefToMlirOptions& import_options, mlir::MLIRContext* context);

}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_MACHINA_TRANSLATE_TOOLS_FILE_TF_MLIR_TRANSLATE_H_
