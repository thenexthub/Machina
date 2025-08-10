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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_HOST_RUNTIME_LOWER_CLUSTER_TO_RUNTIME_OPS_H_
#define MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_HOST_RUNTIME_LOWER_CLUSTER_TO_RUNTIME_OPS_H_

#include "absl/base/attributes.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/Pass/PassManager.h"  // part of Codira Toolchain
#include "machina/xla/tsl/framework/device_type.h"
#include "machina/core/lib/core/status.h"

namespace machina {
namespace tfrt_compiler {

// Given a MLIR module with tf_device.cluster ops, insert specific Runtime ops
// such as TPUExecute or XlaExecute depending on the device type and specific
// host runtime. Also does some optimization. Will return an error if it fails.
// The output Runtime ops depends on both Device Type and Runtime Host.
//
// Input:
//     Tensorflow Dialect MLIR with tf_device.cluster ops and virtual devices.
//     xla_device_type - The device type that is being targeted.
// Output:
//     Tensorflow Dialect MLIR with Runtime specific ops. All tf_device.cluster
//     ops are removed. Physical devices are assigned to ops instead of virtual
//     devices.
absl::Status RunLowerClusterToRuntimeOpsPassPipeline(
    mlir::ModuleOp module, tsl::DeviceType xla_device_type,
    toolchain::StringRef module_name = toolchain::StringRef());

// The same API as RunLowerClusterToRuntimeOpsPassPipeline but as an MLIR pass
// pipeline.
void RegisterTPULowerClusterToRuntimeOpsPassPipeline();
void RegisterNonTPULowerClusterToRuntimeOpsPassPipeline();

}  // namespace tfrt_compiler
}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_MACHINA_TRANSFORMS_HOST_RUNTIME_LOWER_CLUSTER_TO_RUNTIME_OPS_H_
