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

#include "machina/compiler/mlir/tf2xla/internal/legalize_tf_mlir.h"

#include <memory>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "machina/compiler/mlir/machina/dialect_registration.h"
#include "machina/compiler/mlir/machina/transforms/set_tpu_infeed_layout.h"
#include "machina/compiler/mlir/machina/utils/serialize_mlir_module_utils.h"
#include "machina/compiler/mlir/tf2xla/api/v1/compile_mlir_util.h"
#include "machina/compiler/mlir/tf2xla/internal/compilation_timer.h"
#include "machina/compiler/tf2xla/layout_util.h"
#include "machina/compiler/tf2xla/xla_compiler.h"
#include "machina/compiler/tf2xla/xla_helpers.h"
#include "machina/xla/mlir_hlo/mhlo/IR/register.h"
#include "machina/xla/shape.h"
#include "machina/xla/tsl/platform/errors.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/profile_utils/cpu_utils.h"
#include "machina/core/platform/types.h"
#include "machina/core/protobuf/tpu/compile_metadata.pb.h"
#include "machina/core/tpu/kernels/tpu_compile_op_support.h"
#include "machina/core/tpu/tpu_compile.h"
#include "tsl/platform/error_logging.h"

namespace machina {
namespace tf2xla {
namespace internal {

// Name of component for error logging. This name is fixed and required to
// enable logging.
constexpr char kBridgeComponent[] = "TFXLABridge";

using tpu::ShardingAndIndex;

absl::Status CompileFromMlirToXlaHlo(
    bool lower_to_xla_hlo, mlir::ModuleOp mlir_module_op,
    const tpu::TPUCompileMetadataProto& metadata, toolchain::StringRef device_type,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns& shape_determination_fns,
    bool use_tuple_args, XlaCompiler::CompilationResult* compilation_result,
    std::vector<std::unique_ptr<mlir::Pass>>& custom_legalization_passes,
    const std::vector<TensorShape>& arg_shapes,
    std::vector<ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes) {
  LOG_FIRST_N(INFO, 1)
      << "Compiling MLIR computation to XLA HLO using MLIR tf2xla bridge in "
         "the op by op fallback mode. This is Phase 2 of the TF2XLA Bridge. "
         "Old (non-MLIR) bridge may be used in case of unsupported feature "
         "or compilation failure from the MLIR bridge (full fallback mode).";

  toolchain::SmallVector<TensorOrResourceShape, 4> tensor_or_resource_shapes;
  tensor_or_resource_shapes.reserve(arg_shapes.size());
  for (const auto& arg_shape : arg_shapes)
    tensor_or_resource_shapes.push_back({arg_shape});

  TF_RETURN_IF_ERROR(CompileMlirToXlaHlo(
      mlir_module_op, tensor_or_resource_shapes, device_type, use_tuple_args,
      /*enable_op_fallback=*/true, /*use_return_tuple=*/true,
      /*use_resource_updates_for_aliases=*/false, shape_determination_fns,
      compilation_result, custom_legalization_passes, metadata.module_name(),
      lower_to_xla_hlo));

  // Compute how arguments are shared across different cores.
  auto sharding_result =
      tpu::GetShardingInfo(metadata, arg_shapes, shape_determination_fns,
                           arg_core_mapping, per_core_arg_shapes);
  if (!sharding_result.ok()) {
    return sharding_result;
  }
  return absl::OkStatus();
}

};  // namespace internal
};  // namespace tf2xla
};  // namespace machina
