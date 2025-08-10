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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_UTILS_DEVICE_UTIL_H_
#define MACHINA_COMPILER_MLIR_MACHINA_UTILS_DEVICE_UTIL_H_

#include "toolchain/ADT/SmallVector.h"
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_structs.h"
#include "machina/core/common_runtime/device_set.h"
#include "machina/core/util/device_name_utils.h"

namespace machina {

// Collects all devices known to the system by name and adds them as a
// `tf.devices` dictionary attribute with a full device name as a key, and
// device metadata as a value.
//
// Device names added in full parsed device form:
//   /job:<name>/replica:<replica>/task:<task>/device:<type>:<device_num>
//
// Supported device metadata types:
// (1) GpuDeviceMetadata: GPU device compute capability.
void AddDevicesToOp(mlir::Operation* op, const DeviceSet* device_set);

// Collects devices information from an op `tf.devices` attributes. Returns
// failure if can't parse device metadata from the attribute.
mlir::LogicalResult GetDevicesFromOp(mlir::Operation* op,
                                     mlir::TF::RuntimeDevices* devices);

// Parses a device string and returns its ordinal (id). This will return an
// error if the device string is invalid or has no id.
mlir::LogicalResult GetDeviceOrdinalFromDeviceString(mlir::Location loc,
                                                     toolchain::StringRef device,
                                                     int64_t* device_ordinal);

}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_MACHINA_UTILS_DEVICE_UTIL_H_
