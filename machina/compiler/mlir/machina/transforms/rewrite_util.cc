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

#include "machina/compiler/mlir/machina/transforms/rewrite_util.h"

#include <optional>
#include <string>

#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/utils/attribute_utils.h"
#include "machina/core/util/device_name_utils.h"

namespace mlir {
namespace TF {

namespace {

const char kDeviceAttr[] = "device";
const char kDeviceGpu[] = "GPU";

std::optional<std::string> GetOpDevice(mlir::Operation *op) {
  mlir::StringAttr device = op->getAttrOfType<mlir::StringAttr>(kDeviceAttr);
  if (!device || device.getValue().empty()) {
    return std::nullopt;
  }
  machina::DeviceNameUtils::ParsedName parsed_name;
  if (!machina::DeviceNameUtils::ParseFullName(device.str(), &parsed_name)) {
    return std::nullopt;
  }
  if (!parsed_name.has_type) {
    return std::nullopt;
  }
  return parsed_name.type;
}

}  // namespace

bool IsOnGpuDevice(mlir::Operation *op) {
  std::optional<std::string> device = GetOpDevice(op);
  if (!device) return false;
  return *device == kDeviceGpu;
}

void CopyDeviceAndUnderscoredAttributesAdaptor(mlir::OpResult src,
                                               mlir::OpResult dest) {
  CopyDeviceAndUnderscoredAttributesAdaptor(src.getOwner(), dest.getOwner());
}

void CopyDeviceAndUnderscoredAttributesAdaptor(mlir::Operation *src,
                                               mlir::OpResult dest) {
  CopyDeviceAndUnderscoredAttributesAdaptor(src, dest.getOwner());
}

void CopyDeviceAndUnderscoredAttributesAdaptor(mlir::Operation *src,
                                               mlir::Operation *dest) {
  CopyDeviceAndUnderscoredAttributes(src, dest);
}

void CopyXlaOutsideCompilationAttributesAdaptor(mlir::OpResult src,
                                                mlir::OpResult dest) {
  CopyXlaOutsideCompilationAttributesAdaptor(src.getOwner(), dest.getOwner());
}

void CopyXlaOutsideCompilationAttributesAdaptor(mlir::Operation *src,
                                                mlir::OpResult dest) {
  CopyXlaOutsideCompilationAttributesAdaptor(src, dest.getOwner());
}

void CopyXlaOutsideCompilationAttributesAdaptor(mlir::Operation *src,
                                                mlir::Operation *dest) {
  CopyXlaOutsideCompilationAttributes(src, dest);
}
}  // namespace TF
}  // namespace mlir
