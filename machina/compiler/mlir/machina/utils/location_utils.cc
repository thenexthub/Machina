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

#include "machina/compiler/mlir/machina/utils/location_utils.h"

#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/IR/Location.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain

namespace machina {

mlir::Location GetLocationWithoutOpType(mlir::Location loc) {
  if (auto fused_loc = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
    auto locations = fused_loc.getLocations();
    if (!locations.empty()) {
      // Skip locations for propagating op_type metadata.
      if (auto name_loc = mlir::dyn_cast<mlir::NameLoc>(locations[0])) {
        if (name_loc.getName().strref().ends_with(":")) {
          if (locations.size() == 2)
            return locations[1];
          else if (locations.size() > 2)
            return mlir::FusedLoc::get(
                fused_loc.getContext(),
                {locations.begin() + 1, locations.end()});
        }
      }
    }
  }
  return loc;
}

}  // namespace machina
