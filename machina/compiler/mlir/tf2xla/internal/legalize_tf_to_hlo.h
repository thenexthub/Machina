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

#ifndef MACHINA_COMPILER_MLIR_TF2MACHINA_XLAINTERNAL_LEGALIZE_TF_TO_HLO_H_
#define MACHINA_COMPILER_MLIR_TF2MACHINA_XLAINTERNAL_LEGALIZE_TF_TO_HLO_H_

#include "absl/status/statusor.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "machina/compiler/tf2xla/xla_helpers.h"
#include "machina/xla/client/compile_only_client.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/core/protobuf/tpu/compile_metadata.pb.h"
#include "machina/core/tpu/kernels/tpu_compile_op_support.h"

namespace machina {
namespace tf2xla {
namespace internal {

// Legalize the given MLIR module to XLA HLO using a combination of the MLIR
// Bridge and XlaBuilder
absl::StatusOr<XlaCompilationResult> LegalizeTfToHlo(
    const tpu::MlirToHloArgs& computation,
    const tpu::TPUCompileMetadataProto& metadata, bool use_tuple_args,
    toolchain::StringRef device_type,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    const std::vector<machina::TensorShape>& arg_shapes,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes,
    std::vector<std::unique_ptr<mlir::Pass>>& custom_legalization_passes,
    xla::CompileOnlyClient* client, XlaCompilationResult* compilation_result);

};  // namespace internal
};  // namespace tf2xla
};  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TF2MACHINA_XLAINTERNAL_LEGALIZE_TF_TO_HLO_H_
