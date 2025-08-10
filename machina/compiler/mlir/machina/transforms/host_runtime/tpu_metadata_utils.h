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
#ifndef MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_HOST_RUNTIME_TPU_METADATA_UTILS_H_
#define MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_HOST_RUNTIME_TPU_METADATA_UTILS_H_

#include <optional>

#include "mlir/IR/Diagnostics.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_device.h"
#include "machina/xla/xla.pb.h"
#include "machina/xla/xla_data.pb.h"
#include "machina/core/framework/tensor_shape.pb.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/protobuf/tpu/compile_metadata.pb.h"

namespace mlir {
namespace TFTPU {

// Populates a TPUCompileMetadataProto from attributes of a
// `tf_device::ClusterFuncOp`. If any necessary attributes are missing from the
// op, a failure will be returned.
// TODO(lyandy): Support session handle and guaranteed consts.
LogicalResult SetMetadataProtoFromClusterFuncOp(
    tf_device::ClusterFuncOp op, int num_replicas, int num_cores_per_replica,
    std::optional<xla::DeviceAssignmentProto>&& xla_device_assignment,
    machina::tpu::TPUCompileMetadataProto* metadata);
}  // namespace TFTPU
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_HOST_RUNTIME_TPU_METADATA_UTILS_H_
