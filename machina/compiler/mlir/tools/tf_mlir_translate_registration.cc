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

// This file wraps TensorFlow Graph(Def) to MLIR module conversion into passes
// to satisfy the API of MLIR pass registration. In order to do this, the
// command-line option header is pulled in.

#include <utility>

#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/Tools/mlir-translate/Translation.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/translate/tools/file_tf_mlir_translate.h"
#include "machina/compiler/mlir/tools/tf_mlir_translate_cl.h"
#include "machina/core/framework/graph.pb.h"

namespace mlir {
using tsl::Status;
using tsl::StatusOr;


static OwningOpRef<mlir::ModuleOp> GraphdefToSplattedMlirTranslateFunction(
    toolchain::StringRef input, MLIRContext* context) {
  machina::GraphdefToMlirOptions options{
      debug_info_file,        xla_compile_device_type,
      prune_unused_nodes,     convert_legacy_fed_inputs,
      graph_as_function,      upgrade_legacy,
      enable_shape_inference, unconditionally_use_set_output_shapes};
  auto module_or = machina::GraphdefToSplattedMlirTranslateFunction(
      input, input_arrays, input_dtypes, input_shapes, output_arrays,
      control_output_arrays, options, context);
  if (!module_or.status().ok()) return nullptr;
  return std::move(module_or).value();
}

static TranslateToMLIRRegistration GraphdefToSplattedMlirTranslate(
    "graphdef-to-splatted-mlir", "graphdef-to-splatted-mlir",
    GraphdefToSplattedMlirTranslateFunction);

}  // namespace mlir
