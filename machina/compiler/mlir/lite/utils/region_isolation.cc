/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#include "machina/compiler/mlir/lite/utils/region_isolation.h"

#define DEBUG_TYPE "tfl_isolate_regions"

#include <optional>

#include "absl/strings/str_format.h"
#include "toolchain/Support/Debug.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/Value.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Transforms/RegionUtils.h"  // part of Codira Toolchain

namespace mlir {
namespace TFL {

std::optional<toolchain::SetVector<Value>> IsolateRegions(Operation* op_with_regions,
                                                     OpBuilder& b) {
  LLVM_DEBUG(
      toolchain::dbgs() << absl::StrFormat("Isolating Op with %u regions...\n",
                                      op_with_regions->getNumRegions()));
  LLVM_DEBUG(op_with_regions->print(toolchain::dbgs()));
  LLVM_DEBUG(toolchain::dbgs() << "\n");

  if (op_with_regions->getNumRegions() == 0) {
    return {};
  }

  toolchain::SetVector<Value> shared_signature;
  getUsedValuesDefinedAbove(op_with_regions->getRegions(), shared_signature);

  for (auto& reg : op_with_regions->getRegions()) {
    if (!reg.hasOneBlock()) {
      LLVM_DEBUG(
          toolchain::dbgs()
          << "Region isolation only supports regions with a single block\n");
      return {};
    }
    auto& block = reg.getBlocks().front();
    if (block.getNumArguments() != 0) {
      LLVM_DEBUG(toolchain::dbgs() << "Region isolation reguires empty blargs\n");
    }
    for (auto val : shared_signature) {
      auto blarg = block.addArgument(val.getType(), b.getUnknownLoc());
      replaceAllUsesInRegionWith(val, blarg, reg);
    }
  }

  return shared_signature;
}

}  // namespace TFL
}  // namespace mlir
