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

#ifndef MACHINA_CORE_TPU_TPU_COMPILE_H_
#define MACHINA_CORE_TPU_TPU_COMPILE_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "machina/compiler/jit/shape_inference.h"
#include "machina/compiler/tf2xla/layout_util.h"
#include "machina/compiler/tf2xla/xla_compiler.h"
#include "machina/xla/client/compile_only_client.h"
#include "machina/xla/shape.h"
#include "machina/core/framework/attr_value.pb.h"
#include "machina/core/framework/function.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.h"
#include "machina/core/graph/graph.h"
#include "machina/core/platform/status.h"
#include "machina/core/protobuf/tpu/compile_metadata.pb.h"
#include "machina/core/tpu/kernels/tpu_compile_op_support.h"

namespace machina {
namespace tpu {
namespace internal {

// Performs shape inference on the body of `graph`. Shapes for arguments
// are taken from `metadata` and `arg_shapes`.
absl::Status RunShapeInferenceOnComputation(
    const tpu::TPUCompileMetadataProto& metadata,
    const std::vector<PartialTensorShape>& arg_shapes, Graph* graph,
    FunctionLibraryRuntime* flr, GraphShapeInfo* shape_info);
}  // namespace internal

// Converts a TF Function into XLA HLO, stores generated HLO module and
// accompanying metadata in CompilationResult.
absl::Status CompileTFFunctionToHlo(
    const FunctionLibraryDefinition& flib_def, int graph_def_version,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    const std::vector<TensorShape>& arg_shapes, const DeviceType& device_type,
    const GuaranteedConsts& guaranteed_constants, const NameAttrList& function,
    const tpu::TPUCompileMetadataProto& metadata,
    xla::CompileOnlyClient* client,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes,
    bool use_tuple_args, XlaCompiler::CompilationResult* compilation_result);

// Gets information regarding how input arguments are sharded across multiple
// cores.
absl::Status GetShardingInfo(
    const tpu::TPUCompileMetadataProto& metadata,
    absl::Span<const TensorShape> arg_shapes,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes);

}  // namespace tpu
}  // namespace machina

#endif  // MACHINA_CORE_TPU_TPU_COMPILE_H_
