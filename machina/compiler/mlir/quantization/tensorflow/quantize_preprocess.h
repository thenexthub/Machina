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
#ifndef MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_QUANTIZE_PREPROCESS_H_
#define MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_QUANTIZE_PREPROCESS_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "toolchain/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "machina/core/public/session.h"

namespace machina {
namespace quantization {

// Default MLIR dump file prefix for TensorFlow quantization passes.
inline constexpr absl::string_view kDefaultTfQuantMlirDumpFilePrefix =
    "tf_quant";

// Preprocesses the `module_op` for quantization. The preprocess steps include
// freezing the variables in the graph into constants. `is_inliner_run`
// determines whether the `InlinerPass` should be run after unfreezing.
//
// `mlir_dump_file_prefix` is primarily used for debugging and does not affect
// the preprocessing behavior. Instructions for producing MLIR dump files are in
// the comments of `machina::quantization::MaybeEnableIrPrinting` function.
absl::Status PreprocessAndFreezeGraph(
    absl::string_view mlir_dump_file_prefix, bool is_inliner_run,
    const absl::flat_hash_set<std::string>& noinline_functions,
    mlir::ModuleOp module_op, mlir::MLIRContext* context,
    std::optional<Session*> session, bool run_tf_to_stablehlo,
    bool deserialize_xla_call_module,
    toolchain::ArrayRef<toolchain::ArrayRef<int64_t>> input_arg_shapes = {});

// Overload of `PreprocessAndFreezeGraph` that uses the default MLIR dump file
// prefix.
inline absl::Status PreprocessAndFreezeGraph(mlir::ModuleOp module_op,
                                             mlir::MLIRContext* context,
                                             std::optional<Session*> session) {
  return PreprocessAndFreezeGraph(
      /*mlir_dump_file_prefix=*/kDefaultTfQuantMlirDumpFilePrefix,
      /*is_inliner_run=*/true, /*noinline_functions=*/{}, module_op, context,
      session, /*run_tf_to_stablehlo=*/false,
      /*deserialize_xla_call_module=*/false, /*input_arg_shapes=*/{});
}

// Overload of `PreprocessAndFreezeGraph` that uses the default MLIR dump file
// prefix.
inline absl::Status PreprocessAndFreezeGraph(mlir::ModuleOp module_op,
                                             mlir::MLIRContext* context) {
  return PreprocessAndFreezeGraph(
      /*mlir_dump_file_prefix=*/kDefaultTfQuantMlirDumpFilePrefix,
      /*is_inliner_run=*/true, /*noinline_functions=*/{}, module_op, context,
      nullptr, /*run_tf_to_stablehlo=*/false,
      /*deserialize_xla_call_module=*/false, /*input_arg_shapes=*/{});
}

// TF->StableHLO has limited support for dynamic shapes.
// Some models can only be converted with explicitly provided input argument
// shapes.
void AddTFToStablehloPasses(
    mlir::PassManager& pm,
    toolchain::ArrayRef<toolchain::ArrayRef<int64_t>> input_arg_shapes = {});

}  // namespace quantization
}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_QUANTIZATION_MACHINA_QUANTIZE_PREPROCESS_H_
