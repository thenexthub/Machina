/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

// This file defines the types used in the standard MLIR TensorFlow dialect.

#ifndef MACHINA_COMPILER_MLIR_MACHINA_IR_TF_STRUCTS_H_
#define MACHINA_COMPILER_MLIR_MACHINA_IR_TF_STRUCTS_H_

#include <optional>

#include "toolchain/ADT/StringMap.h"
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Diagnostics.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "machina/core/ir/types/dialect.h"
#include "machina/core/util/device_name_utils.h"

namespace mlir {
namespace TF {

using GpuDeviceMetadata = tf_type::GpuDeviceMetadataAttr;

// Tensorflow devices available at runtime with corresponding metadata if it is
// available. It's completely valid to have a device without any metadata
// attached to it.
class RuntimeDevices {
  using DeviceNameUtils = ::machina::DeviceNameUtils;
  using ParsedName = ::machina::DeviceNameUtils::ParsedName;

 public:
  // Adds a device with and empty metadata. Device can be of any type.
  void AddDevice(const ParsedName& device);

  // Adds a GPU device with GPU specific metadata.
  void AddGpuDevice(const ParsedName& device,
                    const GpuDeviceMetadata& metadata);

  toolchain::ArrayRef<ParsedName> device_names() const { return device_names_; }
  size_t NumDevices() const { return device_names_.size(); }

  // Returns GPU device metadata if it is available, otherwise returns None.
  std::optional<GpuDeviceMetadata> GetGpuDeviceMetadata(
      const ParsedName& device) const;

 private:
  toolchain::SmallVector<ParsedName, 8> device_names_;
  // TODO(ezhulenev): Add DenseMapInfo<ParsedName> specialization to be able to
  // use ParsedName as a key in a DenseMap.
  toolchain::StringMap<GpuDeviceMetadata> gpu_metadata_;
};

}  // namespace TF
}  // namespace mlir

#endif  // MACHINA_COMPILER_MLIR_MACHINA_IR_TF_STRUCTS_H_
