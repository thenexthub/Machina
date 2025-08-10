/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_TF2MACHINA_MACHINA_XLA_INTERNAL_LEGALIZE_TF_MLIR_H_
#define MACHINA_COMPILER_MLIR_TF2MACHINA_MACHINA_XLA_INTERNAL_LEGALIZE_TF_MLIR_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "machina/compiler/tf2xla/xla_helpers.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/core/protobuf/tpu/compile_metadata.pb.h"
#include "machina/core/tpu/kernels/tpu_compile_op_support.h"

namespace machina {
namespace tf2xla {
namespace internal {

// Runs all the MLIR Bridge passes on the given MLIR module.
// If compile_to_xla_hlo is true then those passes include all the Legalization
// to XLA HLO which is returned in the compilation_result.
absl::Status CompileFromMlirToXlaHlo(
    bool lower_to_xla_hlo, mlir::ModuleOp mlir_module_op,
    const tpu::TPUCompileMetadataProto& metadata, toolchain::StringRef device_type,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns& shape_determination_fns,
    bool use_tuple_args, XlaCompiler::CompilationResult* compilation_result,
    std::vector<std::unique_ptr<mlir::Pass>>& custom_legalization_passes,
    const std::vector<TensorShape>& arg_shapes,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes);

};  // namespace internal
};  // namespace tf2xla
};  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TF2MACHINA_MACHINA_XLA_INTERNAL_LEGALIZE_TF_MLIR_H_
