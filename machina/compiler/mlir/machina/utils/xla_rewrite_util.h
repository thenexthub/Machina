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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_UTILS_MACHINA_MACHINA_XLA_REWRITE_UTIL_H_
#define MACHINA_COMPILER_MLIR_MACHINA_UTILS_MACHINA_MACHINA_XLA_REWRITE_UTIL_H_

#include <optional>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_device.h"
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_structs.h"
#include "machina/compiler/mlir/machina/ir/tf_types.h"
#include "machina/xla/xla_data.pb.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/statusor.h"
#include "machina/core/util/device_name_utils.h"

namespace machina {
// Erase rewritten ClusterFuncOp(s). If TPUPartitionedInputV2Op /
// TPUPartitionedOutputV2Op are present, they must be removed along with the
// ClusterFuncOp(s).
mlir::LogicalResult EraseClusterFuncs(
    toolchain::MutableArrayRef<mlir::tf_device::ClusterFuncOp> to_be_erased);

// Move child processes of the ParallelExecute that do not change. These are all
// children except for the child with the ClusterFunc.
// Returns the index of the child with the ClusterFunc.
int MovePreservedParallelExecuteChildren(
    int num_cores_per_replica,
    toolchain::SmallVector<mlir::Type, 8>& concatenated_output_types,
    mlir::OpBuilder* builder, mlir::tf_device::ClusterFuncOp cluster_func,
    mlir::tf_device::ParallelExecuteOp old_parallel_execute,
    mlir::tf_device::ParallelExecuteOp* new_parallel_execute);

// Wraps single op in `tf_device.launch` for explicit device assignment.
mlir::tf_device::LaunchOp WrapOpInLaunch(mlir::OpBuilder* builder,
                                         mlir::Location loc,
                                         mlir::Operation* op,
                                         toolchain::StringRef device);

}  // namespace machina
#endif  // MACHINA_COMPILER_MLIR_MACHINA_UTILS_MACHINA_MACHINA_XLA_REWRITE_UTIL_H_
